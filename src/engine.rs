//! OCR 引擎
//!
//! OCR Engine
//!
//! 提供完整的 OCR 流程封装，一次调用完成检测和识别

use image::DynamicImage;
use std::path::Path;

use crate::det::{DetModel, DetOptions};
use crate::error::OcrResult;
use crate::mnn::{Backend, InferenceConfig, PrecisionMode};
use crate::postprocess::TextBox;
use crate::rec::{RecModel, RecOptions, RecognitionResult};

/// OCR 结果
#[derive(Debug, Clone)]
pub struct OcrResult_ {
    /// 识别的文本
    pub text: String,
    /// 置信度
    pub confidence: f32,
    /// 边界框
    pub bbox: TextBox,
}

impl OcrResult_ {
    /// 创建新的 OCR 结果
    pub fn new(text: String, confidence: f32, bbox: TextBox) -> Self {
        Self {
            text,
            confidence,
            bbox,
        }
    }
}

/// OCR 引擎配置
#[derive(Debug, Clone)]
pub struct OcrEngineConfig {
    /// 推理后端
    pub backend: Backend,
    /// 线程数
    pub thread_count: i32,
    /// 精度模式
    pub precision_mode: PrecisionMode,
    /// 检测选项
    pub det_options: DetOptions,
    /// 识别选项
    pub rec_options: RecOptions,
    /// 是否启用并行识别（使用 rayon 对多个文本区域并行处理）
    pub enable_parallel: bool,
}

impl Default for OcrEngineConfig {
    fn default() -> Self {
        Self {
            backend: Backend::CPU,
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            det_options: DetOptions::default(),
            rec_options: RecOptions::default(),
            enable_parallel: false, // 默认禁用，避免与 MNN 线程池冲突
        }
    }
}

impl OcrEngineConfig {
    /// 创建新的配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置推理后端
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// 设置线程数
    pub fn with_threads(mut self, threads: i32) -> Self {
        self.thread_count = threads;
        self
    }

    /// 设置精度模式
    pub fn with_precision(mut self, precision: PrecisionMode) -> Self {
        self.precision_mode = precision;
        self
    }

    /// 设置检测选项
    pub fn with_det_options(mut self, options: DetOptions) -> Self {
        self.det_options = options;
        self
    }

    /// 设置识别选项
    pub fn with_rec_options(mut self, options: RecOptions) -> Self {
        self.rec_options = options;
        self
    }

    /// 启用/禁用并行处理
    ///
    /// 注意：当检测到多个文本区域时，使用 rayon 并行识别。
    /// 如果 MNN 已经设置多线程，启用此选项可能导致过多线程竞争。
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// 快速模式预设
    pub fn fast() -> Self {
        Self {
            precision_mode: PrecisionMode::Low,
            det_options: DetOptions::fast(),
            ..Default::default()
        }
    }

    /// 平衡模式预设
    pub fn balanced() -> Self {
        Self {
            det_options: DetOptions::balanced(),
            ..Default::default()
        }
    }

    /// 高精度模式预设
    pub fn high_precision() -> Self {
        Self {
            precision_mode: PrecisionMode::High,
            det_options: DetOptions::high_precision(),
            rec_options: RecOptions::new().with_min_score(0.4),
            ..Default::default()
        }
    }

    /// GPU 模式预设 (Metal)
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn gpu() -> Self {
        Self {
            backend: Backend::Metal,
            ..Default::default()
        }
    }

    /// GPU 模式预设 (OpenCL)
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn gpu() -> Self {
        Self {
            backend: Backend::OpenCL,
            ..Default::default()
        }
    }

    fn to_inference_config(&self) -> InferenceConfig {
        InferenceConfig {
            thread_count: self.thread_count,
            precision_mode: self.precision_mode,
            backend: self.backend,
            ..Default::default()
        }
    }
}

/// OCR 引擎
///
/// 封装完整的 OCR 流程，包括文本检测和识别
///
/// # 示例
///
/// ```ignore
/// use rust_paddle_ocr::{OcrEngine, OcrEngineConfig};
///
/// // 创建引擎
/// let engine = OcrEngine::new(
///     "det_model.mnn",
///     "rec_model.mnn",
///     "ppocr_keys.txt",
///     None,
/// )?;
///
/// // 识别图像
/// let image = image::open("test.jpg")?;
/// let results = engine.recognize(&image)?;
///
/// for result in results {
///     println!("{}: {:.2}", result.text, result.confidence);
/// }
/// ```
pub struct OcrEngine {
    det_model: DetModel,
    rec_model: RecModel,
    config: OcrEngineConfig,
}

impl OcrEngine {
    /// 从模型文件创建 OCR 引擎
    ///
    /// # 参数
    /// - `det_model_path`: 检测模型文件路径
    /// - `rec_model_path`: 识别模型文件路径
    /// - `charset_path`: 字符集文件路径
    /// - `config`: 可选的引擎配置
    pub fn new(
        det_model_path: impl AsRef<Path>,
        rec_model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        // 优化：直接移动配置，避免多次克隆
        let det_options = config.det_options.clone();
        let rec_options = config.rec_options.clone();

        let det_model = DetModel::from_file(det_model_path, Some(inference_config.clone()))?
            .with_options(det_options);

        let rec_model = RecModel::from_file(rec_model_path, charset_path, Some(inference_config))?
            .with_options(rec_options);

        Ok(Self {
            det_model,
            rec_model,
            config,
        })
    }

    /// 从模型字节创建 OCR 引擎
    pub fn from_bytes(
        det_model_bytes: &[u8],
        rec_model_bytes: &[u8],
        charset_bytes: &[u8],
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<Self> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        // 优化：直接移动配置，避免多次克隆
        let det_options = config.det_options.clone();
        let rec_options = config.rec_options.clone();

        let det_model = DetModel::from_bytes(det_model_bytes, Some(inference_config.clone()))?
            .with_options(det_options);

        let rec_model = RecModel::from_bytes_with_charset(
            rec_model_bytes,
            charset_bytes,
            Some(inference_config),
        )?
        .with_options(rec_options);

        Ok(Self {
            det_model,
            rec_model,
            config,
        })
    }

    /// 只创建检测引擎
    pub fn det_only(
        det_model_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<DetOnlyEngine> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        let det_model = DetModel::from_file(det_model_path, Some(inference_config))?
            .with_options(config.det_options);

        Ok(DetOnlyEngine { det_model })
    }

    /// 只创建识别引擎
    pub fn rec_only(
        rec_model_path: impl AsRef<Path>,
        charset_path: impl AsRef<Path>,
        config: Option<OcrEngineConfig>,
    ) -> OcrResult<RecOnlyEngine> {
        let config = config.unwrap_or_default();
        let inference_config = config.to_inference_config();

        let rec_model = RecModel::from_file(rec_model_path, charset_path, Some(inference_config))?
            .with_options(config.rec_options);

        Ok(RecOnlyEngine { rec_model })
    }

    /// 执行完整的 OCR 识别
    ///
    /// # 参数
    /// - `image`: 输入图像
    ///
    /// # 返回
    /// OCR 结果列表，每个结果包含文本、置信度和边界框
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<Vec<OcrResult_>> {
        // 1. 检测文本区域
        let detections = self.det_model.detect_and_crop(image)?;

        if detections.is_empty() {
            return Ok(Vec::new());
        }

        // 2. 批量识别（避免克隆）
        let (images, boxes): (Vec<&DynamicImage>, Vec<TextBox>) = detections
            .iter()
            .map(|(img, bbox)| (img, bbox.clone()))
            .unzip();

        let rec_results = if self.config.enable_parallel && images.len() > 4 {
            // 并行识别：对于多个文本区域，使用 rayon 并行处理
            use rayon::prelude::*;
            images
                .par_iter()
                .map(|img| self.rec_model.recognize(img))
                .collect::<OcrResult<Vec<_>>>()?
        } else {
            // 序列识别：使用批量推理
            self.rec_model.recognize_batch_ref(&images)?
        };

        // 3. 组合结果
        let results: Vec<OcrResult_> = rec_results
            .into_iter()
            .zip(boxes)
            .filter(|(rec, _)| !rec.text.is_empty())
            .map(|(rec, bbox)| OcrResult_::new(rec.text, rec.confidence, bbox))
            .collect();

        Ok(results)
    }

    /// 只执行检测
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.det_model.detect(image)
    }

    /// 只执行识别 (需要预先裁剪好的文本行图像)
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        self.rec_model.recognize(image)
    }

    /// 批量识别文本行图像
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        self.rec_model.recognize_batch(images)
    }

    /// 获取检测模型引用
    pub fn det_model(&self) -> &DetModel {
        &self.det_model
    }

    /// 获取识别模型引用
    pub fn rec_model(&self) -> &RecModel {
        &self.rec_model
    }

    /// 获取配置
    pub fn config(&self) -> &OcrEngineConfig {
        &self.config
    }
}

/// 只有检测功能的引擎
pub struct DetOnlyEngine {
    det_model: DetModel,
}

impl DetOnlyEngine {
    /// 检测图像中的文本区域
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.det_model.detect(image)
    }

    /// 检测并返回裁剪后的图像
    pub fn detect_and_crop(&self, image: &DynamicImage) -> OcrResult<Vec<(DynamicImage, TextBox)>> {
        self.det_model.detect_and_crop(image)
    }

    /// 获取检测模型引用
    pub fn model(&self) -> &DetModel {
        &self.det_model
    }
}

/// 只有识别功能的引擎
pub struct RecOnlyEngine {
    rec_model: RecModel,
}

impl RecOnlyEngine {
    /// 识别单张图像
    pub fn recognize(&self, image: &DynamicImage) -> OcrResult<RecognitionResult> {
        self.rec_model.recognize(image)
    }

    /// 只返回文本
    pub fn recognize_text(&self, image: &DynamicImage) -> OcrResult<String> {
        self.rec_model.recognize_text(image)
    }

    /// 批量识别
    pub fn recognize_batch(&self, images: &[DynamicImage]) -> OcrResult<Vec<RecognitionResult>> {
        self.rec_model.recognize_batch(images)
    }

    /// 获取识别模型引用
    pub fn model(&self) -> &RecModel {
        &self.rec_model
    }
}

/// 便捷函数：从文件识别
///
/// # 示例
///
/// ```ignore
/// let results = rust_paddle_ocr::ocr_file(
///     "test.jpg",
///     "det_model.mnn",
///     "rec_model.mnn",
///     "ppocr_keys.txt",
/// )?;
/// ```
pub fn ocr_file(
    image_path: impl AsRef<Path>,
    det_model_path: impl AsRef<Path>,
    rec_model_path: impl AsRef<Path>,
    charset_path: impl AsRef<Path>,
) -> OcrResult<Vec<OcrResult_>> {
    let image = image::open(image_path)?;
    let engine = OcrEngine::new(det_model_path, rec_model_path, charset_path, None)?;
    engine.recognize(&image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_config() {
        let config = OcrEngineConfig::default();
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.backend, Backend::CPU);

        let config = OcrEngineConfig::fast();
        assert_eq!(config.precision_mode, PrecisionMode::Low);

        let config = OcrEngineConfig::high_precision();
        assert_eq!(config.precision_mode, PrecisionMode::High);
    }

    #[test]
    fn test_ocr_result() {
        let bbox = TextBox::new(imageproc::rect::Rect::at(0, 0).of_size(100, 20), 0.9);
        let result = OcrResult_::new("Hello".to_string(), 0.95, bbox);

        assert_eq!(result.text, "Hello");
        assert_eq!(result.confidence, 0.95);
    }
}
