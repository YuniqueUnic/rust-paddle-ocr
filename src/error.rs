//! OCR 错误类型定义
//!
//! OCR Error Type Definitions

use thiserror::Error;

use crate::mnn::MnnError;

/// OCR 错误类型
#[derive(Error, Debug)]
pub enum OcrError {
    /// MNN 推理引擎错误
    #[error("MNN 推理错误: {0}")]
    MnnError(#[from] MnnError),

    /// 图像处理错误
    #[error("图像处理错误: {0}")]
    ImageError(#[from] image::ImageError),

    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),

    /// 无效参数错误
    #[error("无效参数: {0}")]
    InvalidParameter(String),

    /// 模型加载错误
    #[error("模型加载失败: {0}")]
    ModelLoadError(String),

    /// 预处理错误
    #[error("预处理错误: {0}")]
    PreprocessError(String),

    /// 后处理错误
    #[error("后处理错误: {0}")]
    PostprocessError(String),

    /// 检测错误
    #[error("检测错误: {0}")]
    DetectionError(String),

    /// 识别错误
    #[error("识别错误: {0}")]
    RecognitionError(String),

    /// 未初始化错误
    #[error("未初始化: {0}")]
    NotInitialized(String),

    /// 字符集解析错误
    #[error("字符集解析错误: {0}")]
    CharsetError(String),
}

/// OCR 结果类型别名
pub type OcrResult<T> = std::result::Result<T, OcrError>;
