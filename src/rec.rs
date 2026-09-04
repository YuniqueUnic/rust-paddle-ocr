//! Text Recognition Model
//!
//! Provides text recognition functionality based on PaddleOCR recognition models

use image::{DynamicImage, GenericImageView};
use ndarray::ArrayD;
use std::path::Path;

use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, InferenceEngine};
use crate::preprocess::{preprocess_for_rec, NormalizeParams};

/// Recognition result
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Recognized text
    pub text: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Confidence score for each character
    pub char_scores: Vec<(char, f32)>,
}

impl RecognitionResult {
    /// Create a new recognition result
    pub fn new(text: String, confidence: f32, char_scores: Vec<(char, f32)>) -> Self {
        Self {
            text,
            confidence,
            char_scores,
        }
    }

    /// Check if the result is valid (confidence above threshold)
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Recognition options
#[derive(Debug, Clone)]
pub struct RecOptions {
    /// Target height (recognition model input height)
    pub target_height: u32,
    /// Minimum confidence threshold (characters below this value will be filtered)
    pub min_score: f32,
    /// Minimum confidence threshold for punctuation
    pub punct_min_score: f32,
}

impl Default for RecOptions {
    fn default() -> Self {
        Self {
            target_height: 48,
            min_score: 0.3, // Lower threshold, model output is raw logit
            punct_min_score: 0.1,
        }
    }
}

impl RecOptions {
    /// Create new recognition options
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target height
    pub fn with_target_height(mut self, height: u32) -> Self {
        self.target_height = height;
        self
    }

    /// Set minimum confidence
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// Set punctuation minimum confidence
    pub fn with_punct_min_score(mut self, score: f32) -> Self {
        self.punct_min_score = score;
        self
    }
}

/// Text recognition model
pub struct RecModel {
    engine: InferenceEngine,
    /// Character set (index to character mapping)
    charset: Vec<char>,
    options: RecOptions,
    normalize_params: NormalizeParams,
}

/// Common punctuation marks
const PUNCTUATIONS: [char; 49] = [
    ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\',
    '|', '@', '#', '$', '%', '&', '*', '+', '=', '~', '，', '。', '！', '？', '；', '：', '、',
    '「', '」', '『', '』', '（', '）', '【', '】', '《', '》', '—', '…', '·', '～',
];

impl RecModel {
    /// Create recognizer from model file and charset file
    ///
    /// # Parameters
    /// - `model_path`: Model file path (.mnn format)
    /// - `charset_path`: Charset file path (one character per line)
    /// - `config`: Optional inference config
    pub fn from_file(
        model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_file(model_path, config)?;
        let charset = Self::load_charset_from_file(charset_path)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Create recognizer from model bytes and charset file
    pub fn from_bytes(
        model_bytes: &[u8],
        charset_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        let charset = Self::load_charset_from_file(charset_path)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Create recognizer from model bytes and charset bytes
    pub fn from_bytes_with_charset(
        model_bytes: &[u8],
        charset_bytes: &[u8],
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        let charset = Self::parse_charset(charset_bytes)?;

        Ok(Self {
            engine,
            charset,
            options: RecOptions::default(),
            normalize_params: NormalizeParams::paddle_rec(),
        })
    }

    /// Load charset from file
    fn load_charset_from_file(path: impl AsRef<Path>) -> OcrResult<Vec<char>> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_charset(content.as_bytes())
    }

    /// Parse charset data
    fn parse_charset(data: &[u8]) -> OcrResult<Vec<char>> {
        let content = std::str::from_utf8(data)
            .map_err(|e| OcrError::CharsetError(format!("UTF-8 decode error: {}", e)))?;

        // Charset format: one character per line
        // Add space at beginning and end as blank and padding
        let mut charset: Vec<char> = vec![' ']; // blank token at start

        for ch in content.chars() {
            if ch != '\n' && ch != '\r' {
                charset.push(ch);
            }
        }

        charset.push(' '); // padding token at end

        if charset.len() < 3 {
            return Err(OcrError::CharsetError("Charset too small".to_string()));
        }

        Ok(charset)
    }

    /// Set recognition options
    pub fn with_options(mut self, options: RecOptions) -> Self {
        self.options = options;
        self
    }

    /// Get current recognition options
    pub fn options(&self) -> &RecOptions {
        &self.options
    }

    /// Modify recognition options
    pub fn options_mut(&mut self) -> &mut RecOptions {
        &mut self.options
    }

    /// Get charset size
    pub fn charset_size(&self) -> usize {
        self.charset.len()
    }

    /// Recognize a single image
    ///
    /// # Parameters
    /// - `image`: Input image (text line image)
    ///
    /// # Returns
    /// Recognition result
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        let input = preprocess_for_rec(image, self.options.target_height, &self.normalize_params)?;

        // Inference writes straight into `logits`, which is the only copy of the model's
        // output that exists. It is large — `seq_len * num_classes` floats, and the charset
        // has tens of thousands of entries — so extra copies are worth avoiding.
        let mut logits = Vec::new();
        let shape = self
            .engine
            .run_dynamic_into(input.view().into_dyn(), &mut logits)?;

        // `preprocess_for_rec` feeds exactly one image, so the output is [seq_len, classes]
        // or [1, seq_len, classes]: the trailing dimension indexes the charset and
        // everything before the last two is a batch dimension of one.
        let num_classes = match *shape.as_slice() {
            [_, classes] if classes > 0 => classes,
            [1, _, classes] if classes > 0 => classes,
            _ => {
                return Err(OcrError::PostprocessError(format!(
                    "Unexpected recognition output shape: {shape:?}"
                )))
            }
        };

        Ok(ctc_decode(
            &logits,
            num_classes,
            &self.charset,
            &self.options,
        ))
    }

    /// Recognize a single image, return text only
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<String> {
        let result = self.recognize(image)?;
        Ok(result.text)
    }

    /// Recognize several text line images.
    ///
    /// Each line is inferred at its own width. Padding a group of lines up to the widest
    /// one measurably costs both accuracy and memory, and buys no speed: the wrapper
    /// serializes MNN inference process-wide, so there is no batch parallelism to gain.
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        let mut out = vec![None; images.len()];
        for i in widest_first(
            images.iter().map(|i| i.dimensions()),
            self.options.target_height,
        ) {
            out[i] = Some(self.recognize(&images[i])?);
        }
        Ok(out.into_iter().map(expect_visited).collect())
    }

    /// Recognize several text line images held by reference.
    pub fn recognize_batch_ref(
        &self,
        images: &[&DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        let mut out = vec![None; images.len()];
        for i in widest_first(
            images.iter().map(|i| i.dimensions()),
            self.options.target_height,
        ) {
            out[i] = Some(self.recognize(images[i])?);
        }
        Ok(out.into_iter().map(expect_visited).collect())
    }

    /// Recognize several text line images, overlapping the per-line preprocessing and
    /// CTC decode across threads. Inference itself is serialized inside the wrapper, so
    /// what threads win here is the decode, which scans `seq_len * num_classes` scores.
    ///
    /// The widest line runs first and on its own: MNN sizes its dynamic memory pool for
    /// the largest input shape it has seen and never shrinks it, so letting one line
    /// establish that high-water keeps the parallel tail — every line of which is
    /// narrower — inside the existing allocation. Fanning all lines out at once instead
    /// measured 20-76 MB higher peak, for no gain in wall time.
    pub fn recognize_batch_parallel(
        &self,
        images: &[DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        use rayon::prelude::*;

        let order = widest_first(
            images.iter().map(|i| i.dimensions()),
            self.options.target_height,
        );
        let Some((&widest, rest)) = order.split_first() else {
            return Ok(Vec::new());
        };

        let mut out = vec![None; images.len()];
        out[widest] = Some(self.recognize(&images[widest])?);
        for (i, result) in rest
            .par_iter()
            .map(|&i| self.recognize(&images[i]).map(|r| (i, r)))
            .collect::<OcrResult<Vec<_>>>()?
        {
            out[i] = Some(result);
        }
        Ok(out.into_iter().map(expect_visited).collect())
    }
}

/// The width `preprocess_for_rec` will scale a `width` x `height` crop to.
fn rec_input_width(width: u32, height: u32, target_height: u32) -> u32 {
    if height == 0 {
        return width;
    }
    (width as f64 * target_height as f64 / height as f64).round() as u32
}

/// Indices of the given `(width, height)` pairs, widest recognition input first.
///
/// MNN grows its dynamic memory pool to fit the largest input shape it has seen and
/// never shrinks it again, so feeding lines in ascending width makes the pool grow step
/// by step and keeps every intermediate size. Starting with the widest line sets that
/// high-water once and lets every narrower line reuse the allocation; measured 3-33%
/// lower peak footprint, the more so the wider the spread. Lines are recognized
/// independently, so the order cannot change any result.
///
/// Ties break on the original index, so the order is deterministic.
fn widest_first(dimensions: impl Iterator<Item = (u32, u32)>, target_height: u32) -> Vec<usize> {
    let mut widths: Vec<(usize, u32)> = dimensions
        .map(|(w, h)| rec_input_width(w, h, target_height))
        .enumerate()
        .collect();
    widths.sort_unstable_by_key(|&(i, w)| (std::cmp::Reverse(w), i));
    widths.into_iter().map(|(i, _)| i).collect()
}

/// Every index is visited exactly once by `widest_first`, so no slot can stay empty.
fn expect_visited(slot: Option<RecognitionResult>) -> RecognitionResult {
    slot.expect("widest_first yields every index")
}

/// Greedy CTC decode over a `[.., seq_len, num_classes]` logit buffer.
///
/// Kept free of the model and the engine so it is testable on its own: the whole
/// recognition postprocess is this one reduction over `logits`.
fn ctc_decode(
    logits: &[f32],
    num_classes: usize,
    charset: &[char],
    options: &RecOptions,
) -> RecognitionResult {
    let mut char_scores = Vec::new();
    let mut prev_idx = usize::MAX;

    for step in logits.chunks_exact(num_classes) {
        let (max_idx, &max_prob) = step
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            // `chunks_exact` never yields an empty chunk for num_classes > 0.
            .expect("non-empty logit step");

        // CTC collapse: drop the blank class (index 0) and repeats of the previous class.
        if max_idx != 0 && max_idx != prev_idx {
            if let Some(&ch) = charset.get(max_idx) {
                // The model output is already a softmax probability; for very large
                // charsets those scores run small, so it is used as the score directly.
                let threshold = if PUNCTUATIONS.contains(&ch) {
                    options.punct_min_score
                } else {
                    options.min_score
                };
                if max_prob >= threshold {
                    char_scores.push((ch, max_prob));
                }
            }
        }

        prev_idx = max_idx;
    }

    let confidence = if char_scores.is_empty() {
        0.0
    } else {
        char_scores.iter().map(|(_, s)| s).sum::<f32>() / char_scores.len() as f32
    };
    let text: String = char_scores.iter().map(|(ch, _)| ch).collect();

    RecognitionResult::new(text, confidence, char_scores)
}

/// Low-level recognition API
impl RecModel {
    /// Raw inference interface
    ///
    /// Execute model inference directly without preprocessing and postprocessing
    ///
    /// # Parameters
    /// - `input`: Preprocessed input tensor [1, 3, H, W]
    ///
    /// # Returns
    /// Model raw output
    pub fn run_raw(&self, input: ndarray::ArrayViewD<f32>) -> OcrResult<ArrayD<f32>> {
        Ok(self.engine.run_dynamic(input)?)
    }

    /// Get model input shape
    pub fn input_shape(&self) -> &[usize] {
        self.engine.input_shape()
    }

    /// Get model output shape
    pub fn output_shape(&self) -> &[usize] {
        self.engine.output_shape()
    }

    /// Get charset
    pub fn charset(&self) -> &[char] {
        &self.charset
    }

    /// Get character by index
    pub fn get_char(&self, index: usize) -> Option<char> {
        self.charset.get(index).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rec_options_default() {
        let opts = RecOptions::default();
        assert_eq!(opts.target_height, 48);
        assert_eq!(opts.min_score, 0.3);
        assert_eq!(opts.punct_min_score, 0.1);
    }

    #[test]
    fn test_rec_options_builder() {
        let opts = RecOptions::new()
            .with_target_height(32)
            .with_min_score(0.6)
            .with_punct_min_score(0.2);

        assert_eq!(opts.target_height, 32);
        assert_eq!(opts.min_score, 0.6);
        assert_eq!(opts.punct_min_score, 0.2);
    }

    #[test]
    fn test_recognition_result_new() {
        let char_scores = vec![
            ('H', 0.99),
            ('e', 0.94),
            ('l', 0.93),
            ('l', 0.95),
            ('o', 0.94),
        ];
        let result = RecognitionResult::new("Hello".to_string(), 0.95, char_scores.clone());

        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.char_scores.len(), 5);
        assert_eq!(result.char_scores[0].0, 'H');
        assert_eq!(result.char_scores[0].1, 0.99);
    }

    #[test]
    fn test_recognition_result_is_valid() {
        let result = RecognitionResult::new(
            "Hello".to_string(),
            0.95,
            vec![
                ('H', 0.99),
                ('e', 0.94),
                ('l', 0.93),
                ('l', 0.95),
                ('o', 0.94),
            ],
        );

        assert!(result.is_valid(0.9));
        assert!(result.is_valid(0.95));
        assert!(!result.is_valid(0.96));
        assert!(!result.is_valid(0.99));
    }

    #[test]
    fn test_recognition_result_empty() {
        let result = RecognitionResult::new(String::new(), 0.0, vec![]);

        assert!(result.text.is_empty());
        assert_eq!(result.confidence, 0.0);
        assert!(!result.is_valid(0.1));
    }

    // `ctc_decode` is the whole recognition postprocess and needs no model, so the
    // decoding rules are asserted directly here rather than through an engine.

    /// Index 0 is the CTC blank, as `parse_charset` arranges it.
    const CHARSET: [char; 5] = [' ', 'a', 'b', ',', '中'];

    /// One timestep whose argmax is `idx`, scoring `score`.
    fn step(idx: usize, score: f32) -> Vec<f32> {
        let mut row = vec![0.0f32; CHARSET.len()];
        row[idx] = score;
        row
    }

    fn decode(steps: &[Vec<f32>], options: &RecOptions) -> RecognitionResult {
        let logits: Vec<f32> = steps.concat();
        ctc_decode(&logits, CHARSET.len(), &CHARSET, options)
    }

    #[test]
    fn ctc_decode_collapses_repeats_and_skips_blank() {
        // a, a(repeat), blank, a(no longer a repeat), b
        let out = decode(
            &[
                step(1, 0.9),
                step(1, 0.9),
                step(0, 0.9),
                step(1, 0.9),
                step(2, 0.9),
            ],
            &RecOptions::default(),
        );
        assert_eq!(out.text, "aab");
    }

    #[test]
    fn ctc_decode_applies_a_separate_threshold_to_punctuation() {
        let options = RecOptions::new()
            .with_min_score(0.5)
            .with_punct_min_score(0.1);
        // 0.3 is below min_score but above punct_min_score, so only the comma survives.
        let out = decode(&[step(1, 0.3), step(3, 0.3)], &options);
        assert_eq!(out.text, ",");
    }

    #[test]
    fn ctc_decode_averages_confidence_over_kept_characters() {
        let out = decode(&[step(1, 0.8), step(2, 0.6)], &RecOptions::default());
        assert_eq!(out.text, "ab");
        assert!((out.confidence - 0.7).abs() < 1e-6, "{}", out.confidence);
    }

    #[test]
    fn ctc_decode_returns_empty_for_an_all_blank_sequence() {
        let out = decode(&[step(0, 0.9), step(0, 0.9)], &RecOptions::default());
        assert!(out.text.is_empty());
        assert_eq!(out.confidence, 0.0);
        assert!(out.char_scores.is_empty());
    }

    #[test]
    fn ctc_decode_ignores_classes_the_charset_does_not_cover() {
        // A model with more output classes than the charset file lists must not panic;
        // the uncovered class is simply dropped.
        let wider = CHARSET.len() + 2;
        let mut logits = vec![0.0f32; wider * 2];
        logits[wider + (wider - 1)] = 0.9; // second step peaks on an out-of-charset class
        logits[1] = 0.9; // first step peaks on 'a'
        let out = ctc_decode(&logits, wider, &CHARSET, &RecOptions::default());
        assert_eq!(out.text, "a");
    }

    #[test]
    fn ctc_decode_keeps_non_ascii_characters() {
        let out = decode(&[step(4, 0.9)], &RecOptions::default());
        assert_eq!(out.text, "中");
    }

    #[test]
    fn rec_input_width_scales_to_the_target_height() {
        assert_eq!(rec_input_width(200, 100, 48), 96);
        assert_eq!(rec_input_width(2231, 39, 48), 2746);
        // A crop already at the target height passes through unchanged.
        assert_eq!(rec_input_width(320, 48, 48), 320);
        // A degenerate crop must not divide by zero.
        assert_eq!(rec_input_width(320, 0, 48), 320);
    }

    #[test]
    fn widest_first_orders_by_recognition_width_not_raw_width() {
        // Raw widths ascend, but index 0 is the tallest crop and so scales down the most:
        // 100x100 -> 48px, 150x50 -> 144px, 200x40 -> 240px.
        let order = widest_first([(100, 100), (150, 50), (200, 40)].into_iter(), 48);
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn widest_first_breaks_ties_on_input_index() {
        let order = widest_first([(100, 48), (300, 48), (100, 48)].into_iter(), 48);
        assert_eq!(order, vec![1, 0, 2]);
    }

    #[test]
    fn widest_first_yields_every_index_exactly_once() {
        let dims = [(10, 5), (900, 30), (44, 44), (1, 1), (700, 100)];
        let mut visited = widest_first(dims.into_iter(), 48);
        assert_eq!(visited.len(), dims.len());
        visited.sort_unstable();
        assert_eq!(visited, (0..dims.len()).collect::<Vec<_>>());
    }

    #[test]
    fn widest_first_handles_an_empty_input() {
        assert!(widest_first(std::iter::empty(), 48).is_empty());
    }
}
