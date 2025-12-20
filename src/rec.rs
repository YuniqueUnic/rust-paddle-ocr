//! 文本识别模型
//!
//! Text Recognition Model
//!
//! 提供基于 PaddleOCR 识别模型的文本识别功能

use image::DynamicImage;
use ndarray::ArrayD;
use std::path::Path;

use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, InferenceEngine};
use crate::preprocess::{preprocess_for_rec, NormalizeParams};

/// 识别结果
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// 识别出的文本
    pub text: String,
    /// 置信度 (0.0 - 1.0)
    pub confidence: f32,
    /// 每个字符的置信度
    pub char_scores: Vec<(char, f32)>,
}

impl RecognitionResult {
    /// 创建新的识别结果
    pub fn new(text: String, confidence: f32, char_scores: Vec<(char, f32)>) -> Self {
        Self {
            text,
            confidence,
            char_scores,
        }
    }

    /// 判断结果是否有效 (置信度高于阈值)
    pub fn is_valid(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// 识别选项
#[derive(Debug, Clone)]
pub struct RecOptions {
    /// 目标高度 (识别模型输入高度)
    pub target_height: u32,
    /// 最小置信度阈值 (低于此值的字符会被过滤)
    pub min_score: f32,
    /// 标点符号的最小置信度阈值
    pub punct_min_score: f32,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否启用批处理
    pub enable_batch: bool,
}

impl Default for RecOptions {
    fn default() -> Self {
        Self {
            target_height: 48,
            min_score: 0.3, // 降低阈值，模型输出是原始 logit
            punct_min_score: 0.1,
            batch_size: 8,
            enable_batch: true,
        }
    }
}

impl RecOptions {
    /// 创建新的识别选项
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置目标高度
    pub fn with_target_height(mut self, height: u32) -> Self {
        self.target_height = height;
        self
    }

    /// 设置最小置信度
    pub fn with_min_score(mut self, score: f32) -> Self {
        self.min_score = score;
        self
    }

    /// 设置标点符号最小置信度
    pub fn with_punct_min_score(mut self, score: f32) -> Self {
        self.punct_min_score = score;
        self
    }

    /// 设置批处理大小
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// 启用/禁用批处理
    pub fn with_batch(mut self, enable: bool) -> Self {
        self.enable_batch = enable;
        self
    }
}

/// 文本识别模型
pub struct RecModel {
    engine: InferenceEngine,
    /// 字符集 (索引到字符的映射)
    charset: Vec<char>,
    options: RecOptions,
    normalize_params: NormalizeParams,
}

/// 常用标点符号
const PUNCTUATIONS: [char; 49] = [
    ',', '.', '!', '?', ';', ':', '"', '\'', '(', ')', '[', ']', '{', '}', '-', '_', '/', '\\',
    '|', '@', '#', '$', '%', '&', '*', '+', '=', '~', '，', '。', '！', '？', '；', '：', '、',
    '「', '」', '『', '』', '（', '）', '【', '】', '《', '》', '—', '…', '·', '～',
];

impl RecModel {
    /// 从模型文件和字符集文件创建识别器
    ///
    /// # 参数
    /// - `model_path`: 模型文件路径 (.mnn 格式)
    /// - `charset_path`: 字符集文件路径 (每行一个字符)
    /// - `config`: 可选的推理配置
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

    /// 从模型字节和字符集文件创建识别器
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

    /// 从模型字节和字符集字节创建识别器
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

    /// 从字符集文件加载字符集
    fn load_charset_from_file(path: impl AsRef<Path>) -> OcrResult<Vec<char>> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_charset(content.as_bytes())
    }

    /// 解析字符集数据
    fn parse_charset(data: &[u8]) -> OcrResult<Vec<char>> {
        let content = std::str::from_utf8(data)
            .map_err(|e| OcrError::CharsetError(format!("UTF-8 解码错误: {}", e)))?;

        // 字符集格式: 每行一个字符
        // 首尾添加空格作为 blank 和 padding
        let mut charset: Vec<char> = vec![' ']; // 开头的 blank token

        for ch in content.chars() {
            if ch != '\n' && ch != '\r' {
                charset.push(ch);
            }
        }

        charset.push(' '); // 结尾的 padding token

        if charset.len() < 3 {
            return Err(OcrError::CharsetError("字符集太小".to_string()));
        }

        Ok(charset)
    }

    /// 设置识别选项
    pub fn with_options(mut self, options: RecOptions) -> Self {
        self.options = options;
        self
    }

    /// 获取当前识别选项
    pub fn options(&self) -> &RecOptions {
        &self.options
    }

    /// 修改识别选项
    pub fn options_mut(&mut self) -> &mut RecOptions {
        &mut self.options
    }

    /// 获取字符集大小
    pub fn charset_size(&self) -> usize {
        self.charset.len()
    }

    /// 识别单张图像
    ///
    /// # 参数
    /// - `image`: 输入图像 (文本行图像)
    ///
    /// # 返回
    /// 识别结果
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        // 预处理
        let input = preprocess_for_rec(image, self.options.target_height, &self.normalize_params);

        // 推理 (使用动态形状)
        let output = self.engine.run_dynamic(input.view().into_dyn())?;

        // 解码
        self.decode_output(&output)
    }

    /// 识别单张图像，只返回文本
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<String> {
        let result = self.recognize(image)?;
        Ok(result.text)
    }

    /// 批量识别图像
    ///
    /// # 参数
    /// - `images`: 输入图像列表
    ///
    /// # 返回
    /// 识别结果列表
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // 对于少量图像，直接逐个处理
        if images.len() <= 2 || !self.options.enable_batch {
            return images.iter().map(|img| self.recognize(img)).collect();
        }

        // 批量处理
        let mut results = Vec::with_capacity(images.len());

        for chunk in images.chunks(self.options.batch_size) {
            let batch_results = self.recognize_batch_internal(chunk)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// 批量识别图像（借用版本，避免克隆）
    ///
    /// # 参数
    /// - `images`: 输入图像引用列表
    ///
    /// # 返回
    /// 识别结果列表
    pub fn recognize_batch_ref(
        &self,
        images: &[&DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // 对于少量图像，直接逐个处理
        if images.len() <= 2 || !self.options.enable_batch {
            return images.iter().map(|img| self.recognize(img)).collect();
        }

        // 批量处理
        let mut results = Vec::with_capacity(images.len());

        for chunk in images.chunks(self.options.batch_size) {
            // 解引用转换为 Vec<DynamicImage>
            let chunk_owned: Vec<DynamicImage> = chunk.iter().map(|img| (*img).clone()).collect();
            let batch_results = self.recognize_batch_internal(&chunk_owned)?;
            results.extend(batch_results);
        }

        Ok(results)
    }

    /// 内部批量识别
    fn recognize_batch_internal(
        &self,
        images: &[DynamicImage],
    ) -> OcrResult<Vec<RecognitionResult>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        // 如果只有一张图像，直接单独处理
        if images.len() == 1 {
            return Ok(vec![self.recognize(&images[0])?]);
        }

        // 批量预处理
        let batch_input = crate::preprocess::preprocess_batch_for_rec(
            images,
            self.options.target_height,
            &self.normalize_params,
        );

        // 批量推理
        let batch_output = self.engine.run_dynamic(batch_input.view().into_dyn())?;

        // 解码每个样本的输出
        let shape = batch_output.shape();
        if shape.len() != 3 {
            return Err(OcrError::PostprocessError(format!(
                "批量推理输出形状错误: {:?}",
                shape
            )));
        }

        let batch_size = shape[0];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // 提取单个样本的输出
            let sample_output = batch_output.slice(ndarray::s![i, .., ..]).to_owned();
            let sample_output_dyn = sample_output.into_dyn();
            let result = self.decode_output(&sample_output_dyn)?;
            results.push(result);
        }

        Ok(results)
    }

    /// 解码模型输出
    fn decode_output(&self, output: &ArrayD<f32>) -> OcrResult<RecognitionResult> {
        let shape = output.shape();

        // 输出形状应该是 [batch, seq_len, num_classes] 或 [seq_len, num_classes]
        let (seq_len, num_classes) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(OcrError::PostprocessError(format!(
                "无效的输出形状: {:?}",
                shape
            )));
        };

        let output_data: Vec<f32> = output.iter().cloned().collect();

        // CTC 解码
        let mut char_scores = Vec::new();
        let mut prev_idx = 0usize;

        for t in 0..seq_len {
            // 找到当前时间步的最大概率字符
            let start = t * num_classes;
            let end = start + num_classes;
            let probs = &output_data[start..end];

            let (max_idx, &max_prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            // CTC 解码规则: 跳过 blank (索引 0) 和重复字符
            if max_idx != 0 && max_idx != prev_idx {
                if max_idx < self.charset.len() {
                    let ch = self.charset[max_idx];

                    // 使用原始 logit 值作为置信度（模型输出已经是 softmax 后的概率）
                    // 对于大字符集，softmax 分数会非常小，所以直接使用 max_prob
                    let score = max_prob;

                    // 只过滤掉非常低置信度的字符
                    let threshold = if Self::is_punctuation(ch) {
                        self.options.punct_min_score
                    } else {
                        self.options.min_score
                    };

                    if score >= threshold {
                        char_scores.push((ch, score));
                    }
                }
            }

            prev_idx = max_idx;
        }

        // 计算平均置信度
        let confidence = if char_scores.is_empty() {
            0.0
        } else {
            char_scores.iter().map(|(_, s)| s).sum::<f32>() / char_scores.len() as f32
        };

        // 提取文本
        let text: String = char_scores.iter().map(|(ch, _)| ch).collect();

        Ok(RecognitionResult::new(text, confidence, char_scores))
    }

    /// 判断是否为标点符号
    fn is_punctuation(ch: char) -> bool {
        PUNCTUATIONS.contains(&ch)
    }
}

/// 底层识别 API
impl RecModel {
    /// 原始推理接口
    ///
    /// 直接执行模型推理，不进行预处理和后处理
    ///
    /// # 参数
    /// - `input`: 预处理后的输入张量 [1, 3, H, W]
    ///
    /// # 返回
    /// 模型原始输出
    pub fn run_raw(&self, input: ndarray::ArrayViewD<f32>) -> OcrResult<ArrayD<f32>> {
        Ok(self.engine.run_dynamic(input)?)
    }

    /// 获取模型输入形状
    pub fn input_shape(&self) -> &[usize] {
        self.engine.input_shape()
    }

    /// 获取模型输出形状
    pub fn output_shape(&self) -> &[usize] {
        self.engine.output_shape()
    }

    /// 获取字符集
    pub fn charset(&self) -> &[char] {
        &self.charset
    }

    /// 根据索引获取字符
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
        assert_eq!(opts.batch_size, 8);
        assert!(opts.enable_batch);
    }

    #[test]
    fn test_rec_options_builder() {
        let opts = RecOptions::new()
            .with_target_height(32)
            .with_min_score(0.6)
            .with_punct_min_score(0.2)
            .with_batch_size(16)
            .with_batch(false);

        assert_eq!(opts.target_height, 32);
        assert_eq!(opts.min_score, 0.6);
        assert_eq!(opts.punct_min_score, 0.2);
        assert_eq!(opts.batch_size, 16);
        assert!(!opts.enable_batch);
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

    #[test]
    fn test_is_punctuation_common() {
        // 英文标点
        assert!(RecModel::is_punctuation(','));
        assert!(RecModel::is_punctuation('.'));
        assert!(RecModel::is_punctuation('!'));
        assert!(RecModel::is_punctuation('?'));
        assert!(RecModel::is_punctuation(';'));
        assert!(RecModel::is_punctuation(':'));
        assert!(RecModel::is_punctuation('"'));
        assert!(RecModel::is_punctuation('\''));
    }

    #[test]
    fn test_is_punctuation_chinese() {
        // 中文标点
        assert!(RecModel::is_punctuation('，'));
        assert!(RecModel::is_punctuation('。'));
        assert!(RecModel::is_punctuation('！'));
        assert!(RecModel::is_punctuation('？'));
        assert!(RecModel::is_punctuation('；'));
        assert!(RecModel::is_punctuation('：'));
        assert!(RecModel::is_punctuation('、'));
        assert!(RecModel::is_punctuation('—'));
        assert!(RecModel::is_punctuation('…'));
    }

    #[test]
    fn test_is_punctuation_brackets() {
        assert!(RecModel::is_punctuation('('));
        assert!(RecModel::is_punctuation(')'));
        assert!(RecModel::is_punctuation('['));
        assert!(RecModel::is_punctuation(']'));
        assert!(RecModel::is_punctuation('{'));
        assert!(RecModel::is_punctuation('}'));
        assert!(RecModel::is_punctuation('「'));
        assert!(RecModel::is_punctuation('」'));
        assert!(RecModel::is_punctuation('《'));
        assert!(RecModel::is_punctuation('》'));
    }

    #[test]
    fn test_is_punctuation_false() {
        // 非标点字符
        assert!(!RecModel::is_punctuation('A'));
        assert!(!RecModel::is_punctuation('z'));
        assert!(!RecModel::is_punctuation('0'));
        assert!(!RecModel::is_punctuation('中'));
        assert!(!RecModel::is_punctuation('文'));
        assert!(!RecModel::is_punctuation(' '));
    }
}
