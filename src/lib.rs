//! # Rust PaddleOCR
//!
//! 基于 PaddleOCR 模型的高性能 OCR 库，使用 MNN 推理框架。
//!
//! A high-performance OCR library based on PaddleOCR models, using the MNN inference framework.
//!
//! ## 版本 2.0 新特性 (Version 2.0 New Features)
//!
//! - **全新 API 设计**: 提供从底层模型到高层 Pipeline 的完整分层 API
//! - **灵活的模型加载**: 支持从文件路径或内存字节加载模型
//! - **可配置的检测参数**: 支持自定义检测阈值、分辨率、精度模式等
//! - **三种精度模式**: 快速、平衡、高精度模式，满足不同场景需求
//! - **GPU 加速**: 支持 Metal、OpenCL、Vulkan 等多种 GPU 后端
//! - **批量处理**: 支持批量文本识别以提高吞吐量
//!
//! ## 快速开始 (Quick Start)
//!
//! ### 简单用法 - 使用高级 API (推荐)
//!
//! ```ignore
//! use ocr_rs::{OcrEngine, OcrEngineConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建 OCR 引擎
//!     let engine = OcrEngine::new(
//!         "models/det_model.mnn",
//!         "models/rec_model.mnn",
//!         "models/ppocr_keys.txt",
//!         None, // 使用默认配置
//!     )?;
//!
//!     // 打开图像并识别
//!     let image = image::open("test.jpg")?;
//!     let results = engine.recognize(&image)?;
//!
//!     for result in results {
//!         println!("文本: {}, 置信度: {:.2}%", result.text, result.confidence * 100.0);
//!         println!("位置: ({}, {})", result.bbox.rect.left(), result.bbox.rect.top());
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### 高级用法 - 使用底层 API
//!
//! ```ignore
//! use ocr_rs::{DetModel, RecModel, DetOptions, DetPrecisionMode};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建检测模型 (高精度模式)
//!     let det = DetModel::from_file("models/det_model.mnn", None)?
//!         .with_options(DetOptions::high_precision());
//!
//!     // 创建识别模型
//!     let rec = RecModel::from_file("models/rec_model.mnn", "models/ppocr_keys.txt", None)?;
//!
//!     // 加载图像
//!     let image = image::open("test.jpg")?;
//!
//!     // 检测并裁剪文本区域
//!     let detections = det.detect_and_crop(&image)?;
//!
//!     // 识别每个文本区域
//!     for (cropped_img, bbox) in detections {
//!         let result = rec.recognize(&cropped_img)?;
//!         println!("位置: ({}, {}), 文本: {}",
//!             bbox.rect.left(), bbox.rect.top(), result.text);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ### 使用精度预设
//!
//! ```ignore
//! use ocr_rs::{OcrEngine, OcrEngineConfig};
//!
//! // 快速模式 - 适合实时处理
//! let config = OcrEngineConfig::fast();
//!
//! // 平衡模式 - 速度与精度的平衡
//! let config = OcrEngineConfig::balanced();
//!
//! // 高精度模式 - 多尺度 + 分块检测
//! let config = OcrEngineConfig::high_precision();
//!
//! let engine = OcrEngine::new(det_path, rec_path, charset_path, Some(config))?;
//! ```
//!
//! ### GPU 加速
//!
//! ```ignore
//! use ocr_rs::{OcrEngine, OcrEngineConfig, Backend};
//!
//! let config = OcrEngineConfig::new()
//!     .with_backend(Backend::Metal);  // macOS/iOS
//!     // .with_backend(Backend::OpenCL);  // 跨平台
//!
//! let engine = OcrEngine::new(det_path, rec_path, charset_path, Some(config))?;
//! ```
//!
//! ## 模块结构 (Module Structure)
//!
//! - [`mnn`]: MNN 推理引擎封装，提供底层推理能力
//! - [`det`]: 文本检测模型 ([`DetModel`])，检测图像中的文本区域
//! - [`rec`]: 文本识别模型 ([`RecModel`])，识别文本内容
//! - [`engine`]: 高级 OCR Pipeline ([`OcrEngine`])，一站式 OCR 解决方案
//! - [`preprocess`]: 图像预处理工具，包括归一化、缩放等
//! - [`postprocess`]: 后处理工具，包括 NMS、框合并、排序等
//! - [`error`]: 错误类型定义 ([`OcrError`])
//!
//! ## API 层次
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           OcrEngine (高级 API)           │
//! │    一次调用完成检测和识别                 │
//! ├─────────────────────────────────────────┤
//! │     DetModel      │      RecModel       │
//! │   文本检测模型     │    文本识别模型      │
//! ├─────────────────────────────────────────┤
//! │          InferenceEngine (MNN)          │
//! │            底层推理引擎                  │
//! └─────────────────────────────────────────┘
//! ```
//!
//! ## 支持的模型
//!
//! - **PP-OCRv4**: 稳定版本，兼容性好
//! - **PP-OCRv5**: 推荐版本，支持多语言，精度更高
//! - **PP-OCRv5 FP16**: 高效版本，推理速度更快，内存使用更低

// 核心模块
pub mod det;
pub mod engine;
pub mod error;
pub mod mnn;
pub mod postprocess;
pub mod preprocess;
pub mod rec;

// 重新导出常用类型
pub use det::{DetModel, DetOptions, DetPrecisionMode};
pub use engine::{ocr_file, DetOnlyEngine, OcrEngine, OcrEngineConfig, OcrResult_, RecOnlyEngine};
pub use error::{OcrError, OcrResult};
pub use mnn::{Backend, InferenceConfig, InferenceEngine, PrecisionMode};
pub use postprocess::TextBox;
pub use rec::{RecModel, RecOptions, RecognitionResult};

/// 获取库版本
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// 获取 MNN 版本
pub fn mnn_version() -> String {
    mnn::get_version()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
