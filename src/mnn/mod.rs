//! MNN 推理引擎 FFI 绑定层
//!
//! MNN Inference Engine FFI Binding Layer
//!
//! 这个模块封装了 MNN C++ 推理框架的底层接口，提供安全的 Rust API。

use ndarray::{ArrayD, ArrayViewD, IxDyn};
use std::ffi::CStr;
use std::ptr::NonNull;

#[allow(non_camel_case_types)]
#[allow(non_upper_case_globals)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/mnn_bindings.rs"));
}

// ============== 错误类型 ==============

/// MNN 相关错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MnnError {
    /// 无效参数
    InvalidParameter(String),
    /// 内存不足
    OutOfMemory,
    /// 运行时错误
    RuntimeError(String),
    /// 不支持的操作
    Unsupported,
    /// 模型加载失败
    ModelLoadFailed(String),
    /// 空指针错误
    NullPointer,
    /// 形状不匹配
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
}

impl std::fmt::Display for MnnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MnnError::InvalidParameter(msg) => write!(f, "无效参数: {}", msg),
            MnnError::OutOfMemory => write!(f, "内存不足"),
            MnnError::RuntimeError(msg) => write!(f, "运行时错误: {}", msg),
            MnnError::Unsupported => write!(f, "不支持的操作"),
            MnnError::ModelLoadFailed(msg) => write!(f, "模型加载失败: {}", msg),
            MnnError::NullPointer => write!(f, "空指针"),
            MnnError::ShapeMismatch { expected, got } => {
                write!(f, "形状不匹配: 期望 {:?}, 实际 {:?}", expected, got)
            }
        }
    }
}

impl std::error::Error for MnnError {}

pub type Result<T> = std::result::Result<T, MnnError>;

// ============== 配置类型 ==============

/// 精度模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum PrecisionMode {
    /// 正常精度
    #[default]
    Normal = 0,
    /// 低精度 (更快)
    Low = 1,
    /// 高精度 (更准确)
    High = 2,
}

/// 数据格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum DataFormat {
    /// NCHW 格式 (Caffe/PyTorch/ONNX)
    #[default]
    NCHW = 0,
    /// NHWC 格式 (TensorFlow)
    NHWC = 1,
    /// 自动检测
    Auto = 2,
}

/// 推理后端类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU 后端
    #[default]
    CPU,
    /// Metal GPU (macOS/iOS)
    Metal,
    /// OpenCL GPU
    OpenCL,
    /// OpenGL GPU
    OpenGL,
    /// Vulkan GPU
    Vulkan,
    /// CUDA GPU (NVIDIA)
    CUDA,
    /// CoreML (macOS/iOS)
    CoreML,
}

/// 推理配置
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// 线程数 (0 表示自动, 默认为 4)
    pub thread_count: i32,
    /// 精度模式
    pub precision_mode: PrecisionMode,
    /// 是否使用缓存
    pub use_cache: bool,
    /// 数据格式
    pub data_format: DataFormat,
    /// 推理后端
    pub backend: Backend,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            thread_count: 4,
            precision_mode: PrecisionMode::Normal,
            use_cache: false,
            data_format: DataFormat::NCHW,
            backend: Backend::CPU,
        }
    }
}

impl InferenceConfig {
    /// 创建新的推理配置
    pub fn new() -> Self {
        Self::default()
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

    /// 设置后端
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// 设置数据格式
    pub fn with_data_format(mut self, format: DataFormat) -> Self {
        self.data_format = format;
        self
    }

    fn to_ffi(&self) -> ffi::MNNR_Config {
        ffi::MNNR_Config {
            thread_count: self.thread_count,
            precision_mode: self.precision_mode as i32,
            use_cache: self.use_cache,
            data_format: self.data_format as i32,
        }
    }
}

// ============== 共享运行时 ==============

/// 共享运行时，用于多个引擎之间共享资源
pub struct SharedRuntime {
    ptr: NonNull<ffi::MNN_SharedRuntime>,
}

impl SharedRuntime {
    /// 创建新的共享运行时
    pub fn new(config: &InferenceConfig) -> Result<Self> {
        let c_config = config.to_ffi();
        let runtime_ptr = unsafe { ffi::mnnr_create_runtime(&c_config) };

        let ptr = NonNull::new(runtime_ptr)
            .ok_or_else(|| MnnError::RuntimeError("创建共享运行时失败".to_string()))?;

        Ok(SharedRuntime { ptr })
    }

    pub(crate) fn as_ptr(&self) -> *mut ffi::MNN_SharedRuntime {
        self.ptr.as_ptr()
    }
}

impl Drop for SharedRuntime {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_runtime(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for SharedRuntime {}
unsafe impl Sync for SharedRuntime {}

// ============== 辅助函数 ==============

fn get_last_error_message(engine: Option<*const ffi::MNN_InferenceEngine>) -> String {
    match engine {
        Some(ptr) => unsafe {
            let c_str = ffi::mnnr_get_last_error(ptr);
            if c_str.is_null() {
                "未知错误".to_string()
            } else {
                CStr::from_ptr(c_str).to_string_lossy().into_owned()
            }
        },
        None => "引擎创建失败".to_string(),
    }
}

// ============== 推理引擎 ==============

/// MNN 推理引擎
///
/// 封装了 MNN 模型的加载和推理功能
pub struct InferenceEngine {
    ptr: NonNull<ffi::MNN_InferenceEngine>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl InferenceEngine {
    /// 从模型字节数据创建推理引擎
    ///
    /// # 参数
    /// - `model_buffer`: 模型文件的字节数据
    /// - `config`: 可选的推理配置
    ///
    /// # 示例
    /// ```ignore
    /// let model_data = std::fs::read("model.mnn")?;
    /// let engine = InferenceEngine::from_buffer(&model_data, None)?;
    /// ```
    pub fn from_buffer(model_buffer: &[u8], config: Option<InferenceConfig>) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter("模型数据为空".to_string()));
        }

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();

        let engine_ptr = unsafe {
            ffi::mnnr_create_engine(
                model_buffer.as_ptr() as *const _,
                model_buffer.len(),
                &c_config,
            )
        };

        let ptr = NonNull::new(engine_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

        let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

        Ok(InferenceEngine {
            ptr,
            input_shape,
            output_shape,
        })
    }

    /// 从模型文件创建推理引擎
    pub fn from_file(
        model_path: impl AsRef<std::path::Path>,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        let model_buffer = std::fs::read(model_path.as_ref())
            .map_err(|e| MnnError::ModelLoadFailed(format!("读取模型文件失败: {}", e)))?;
        Self::from_buffer(&model_buffer, config)
    }

    /// 使用共享运行时从模型字节数据创建推理引擎
    pub fn from_buffer_with_runtime(model_buffer: &[u8], runtime: &SharedRuntime) -> Result<Self> {
        if model_buffer.is_empty() {
            return Err(MnnError::InvalidParameter("模型数据为空".to_string()));
        }

        let engine_ptr = unsafe {
            ffi::mnnr_create_engine_with_runtime(
                model_buffer.as_ptr() as *const _,
                model_buffer.len(),
                runtime.as_ptr(),
            )
        };

        let ptr = NonNull::new(engine_ptr)
            .ok_or_else(|| MnnError::ModelLoadFailed(get_last_error_message(None)))?;

        let (input_shape, output_shape) = unsafe { Self::get_shapes(ptr.as_ptr())? };

        Ok(InferenceEngine {
            ptr,
            input_shape,
            output_shape,
        })
    }

    unsafe fn get_shapes(ptr: *mut ffi::MNN_InferenceEngine) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut input_shape_vec = vec![0usize; 8];
        let mut input_ndims = 0;
        let mut output_shape_vec = vec![0usize; 8];
        let mut output_ndims = 0;

        if ffi::mnnr_get_input_shape(ptr, input_shape_vec.as_mut_ptr(), &mut input_ndims)
            != ffi::MNNR_ErrorCode_MNNR_SUCCESS
        {
            return Err(MnnError::RuntimeError("获取输入形状失败".to_string()));
        }
        input_shape_vec.truncate(input_ndims);

        if ffi::mnnr_get_output_shape(ptr, output_shape_vec.as_mut_ptr(), &mut output_ndims)
            != ffi::MNNR_ErrorCode_MNNR_SUCCESS
        {
            return Err(MnnError::RuntimeError("获取输出形状失败".to_string()));
        }
        output_shape_vec.truncate(output_ndims);

        Ok((input_shape_vec, output_shape_vec))
    }

    /// 获取输入张量形状
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// 获取输出张量形状
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// 执行推理
    ///
    /// # 参数
    /// - `input_data`: 输入数据，形状必须与模型输入形状匹配
    ///
    /// # 返回
    /// 推理结果数组
    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }

        let input_slice = input_data
            .as_slice()
            .ok_or_else(|| MnnError::InvalidParameter("输入数据必须是连续的".to_string()))?;

        let output_size: usize = self.output_shape.iter().product();
        let mut output_buffer = vec![0.0f32; output_size];

        let error_code = unsafe {
            ffi::mnnr_run_inference(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_slice.len(),
                output_buffer.as_mut_ptr(),
                output_buffer.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer)
                    .map_err(|e| MnnError::RuntimeError(format!("创建输出数组失败: {}", e)))
            }
            ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(MnnError::InvalidParameter(
                get_last_error_message(Some(self.ptr.as_ptr())),
            )),
            ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
            ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
            _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                self.ptr.as_ptr(),
            )))),
        }
    }

    /// 执行推理 (使用原始切片)
    ///
    /// 这是一个低级 API，适用于需要最大性能的场景
    pub fn run_raw(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let expected_input: usize = self.input_shape.iter().product();
        let expected_output: usize = self.output_shape.iter().product();

        if input.len() != expected_input {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_input],
                got: vec![input.len()],
            });
        }

        if output.len() != expected_output {
            return Err(MnnError::ShapeMismatch {
                expected: vec![expected_output],
                got: vec![output.len()],
            });
        }

        let error_code = unsafe {
            ffi::mnnr_run_inference(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => Ok(()),
            ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(MnnError::InvalidParameter(
                get_last_error_message(Some(self.ptr.as_ptr())),
            )),
            ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
            _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                self.ptr.as_ptr(),
            )))),
        }
    }

    pub(crate) fn as_ptr(&self) -> NonNull<ffi::MNN_InferenceEngine> {
        self.ptr
    }

    /// 检查模型是否有动态形状 (包含 -1 维度)
    pub fn has_dynamic_shape(&self) -> bool {
        // 当形状包含非常大的值时，说明是动态形状（-1 转换为 usize 后会变成很大的数）
        self.input_shape.iter().any(|&d| d > 100000)
            || self.output_shape.iter().any(|&d| d > 100000)
    }

    /// 执行动态形状推理
    ///
    /// 适用于输入形状在运行时变化的模型（如检测模型）。
    /// 此函数会在运行前调整模型输入张量的形状。
    ///
    /// # 参数
    /// - `input_data`: 输入数据数组
    ///
    /// # 返回
    /// 推理结果数组，形状由模型动态决定
    pub fn run_dynamic(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        let input_shape: Vec<usize> = input_data.shape().to_vec();
        let input_slice = input_data
            .as_slice()
            .ok_or_else(|| MnnError::InvalidParameter("输入数据必须是连续的".to_string()))?;

        let mut output_data: *mut f32 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let mut output_dims = [0usize; 8];
        let mut output_ndims: usize = 0;

        let error_code = unsafe {
            ffi::mnnr_run_inference_dynamic(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_shape.as_ptr(),
                input_shape.len(),
                &mut output_data,
                &mut output_size,
                output_dims.as_mut_ptr(),
                &mut output_ndims,
            )
        };

        if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
            return match error_code {
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                ffi::MNNR_ErrorCode_MNNR_ERROR_UNSUPPORTED => Err(MnnError::Unsupported),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            };
        }

        // Copy output data and free C buffer
        let output_shape: Vec<usize> = output_dims[..output_ndims].to_vec();
        let output_buffer = unsafe {
            let slice = std::slice::from_raw_parts(output_data, output_size);
            let buffer = slice.to_vec();
            ffi::mnnr_free_output(output_data);
            buffer
        };

        ArrayD::from_shape_vec(IxDyn(&output_shape), output_buffer)
            .map_err(|e| MnnError::RuntimeError(format!("创建输出数组失败: {}", e)))
    }

    /// 执行动态形状推理 (使用原始切片)
    ///
    /// 低级 API，调用者负责管理输出缓冲区
    pub fn run_dynamic_raw(
        &self,
        input: &[f32],
        input_shape: &[usize],
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let mut output_data: *mut f32 = std::ptr::null_mut();
        let mut output_size: usize = 0;
        let mut output_dims = [0usize; 8];
        let mut output_ndims: usize = 0;

        let error_code = unsafe {
            ffi::mnnr_run_inference_dynamic(
                self.ptr.as_ptr(),
                input.as_ptr(),
                input_shape.as_ptr(),
                input_shape.len(),
                &mut output_data,
                &mut output_size,
                output_dims.as_mut_ptr(),
                &mut output_ndims,
            )
        };

        if error_code != ffi::MNNR_ErrorCode_MNNR_SUCCESS {
            return match error_code {
                ffi::MNNR_ErrorCode_MNNR_ERROR_INVALID_PARAMETER => Err(
                    MnnError::InvalidParameter(get_last_error_message(Some(self.ptr.as_ptr()))),
                ),
                ffi::MNNR_ErrorCode_MNNR_ERROR_OUT_OF_MEMORY => Err(MnnError::OutOfMemory),
                _ => Err(MnnError::RuntimeError(get_last_error_message(Some(
                    self.ptr.as_ptr(),
                )))),
            };
        }

        // Copy output and free C buffer
        let output_shape = output_dims[..output_ndims].to_vec();
        let output_buffer = unsafe {
            let slice = std::slice::from_raw_parts(output_data, output_size);
            let buffer = slice.to_vec();
            ffi::mnnr_free_output(output_data);
            buffer
        };

        Ok((output_buffer, output_shape))
    }
}

impl Drop for InferenceEngine {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_engine(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for InferenceEngine {}
unsafe impl Sync for InferenceEngine {}

// ============== 会话池 ==============

/// 会话池，用于高并发推理场景
pub struct SessionPool {
    ptr: NonNull<ffi::MNN_SessionPool>,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl SessionPool {
    /// 创建会话池
    ///
    /// # 参数
    /// - `engine`: 推理引擎
    /// - `pool_size`: 池中会话数量
    /// - `config`: 可选的推理配置
    pub fn new(
        engine: &InferenceEngine,
        pool_size: usize,
        config: Option<InferenceConfig>,
    ) -> Result<Self> {
        if pool_size == 0 {
            return Err(MnnError::InvalidParameter("池大小不能为0".to_string()));
        }

        let cfg = config.unwrap_or_default();
        let c_config = cfg.to_ffi();

        let pool_ptr = unsafe {
            ffi::mnnr_create_session_pool(engine.as_ptr().as_ptr(), pool_size, &c_config)
        };

        let ptr = NonNull::new(pool_ptr)
            .ok_or_else(|| MnnError::RuntimeError("创建会话池失败".to_string()))?;

        Ok(SessionPool {
            ptr,
            input_shape: engine.input_shape.clone(),
            output_shape: engine.output_shape.clone(),
        })
    }

    /// 执行推理 (线程安全)
    pub fn run(&self, input_data: ArrayViewD<f32>) -> Result<ArrayD<f32>> {
        if input_data.shape() != self.input_shape.as_slice() {
            return Err(MnnError::ShapeMismatch {
                expected: self.input_shape.clone(),
                got: input_data.shape().to_vec(),
            });
        }

        let input_slice = input_data
            .as_slice()
            .ok_or_else(|| MnnError::InvalidParameter("输入数据必须是连续的".to_string()))?;

        let output_size: usize = self.output_shape.iter().product();
        let mut output_buffer = vec![0.0f32; output_size];

        let error_code = unsafe {
            ffi::mnnr_session_pool_run(
                self.ptr.as_ptr(),
                input_slice.as_ptr(),
                input_slice.len(),
                output_buffer.as_mut_ptr(),
                output_buffer.len(),
            )
        };

        match error_code {
            ffi::MNNR_ErrorCode_MNNR_SUCCESS => {
                ArrayD::from_shape_vec(IxDyn(&self.output_shape), output_buffer)
                    .map_err(|e| MnnError::RuntimeError(format!("创建输出数组失败: {}", e)))
            }
            _ => Err(MnnError::RuntimeError("会话池推理失败".to_string())),
        }
    }

    /// 获取可用会话数量
    pub fn available(&self) -> usize {
        unsafe { ffi::mnnr_session_pool_available(self.ptr.as_ptr()) }
    }
}

impl Drop for SessionPool {
    fn drop(&mut self) {
        unsafe {
            ffi::mnnr_destroy_session_pool(self.ptr.as_ptr());
        }
    }
}

unsafe impl Send for SessionPool {}
unsafe impl Sync for SessionPool {}

// ============== 工具函数 ==============

/// 获取 MNN 版本号
pub fn get_version() -> String {
    unsafe {
        let c_str = ffi::mnnr_get_version();
        if c_str.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(c_str).to_string_lossy().into_owned()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = InferenceConfig::default();
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.precision_mode, PrecisionMode::Normal);
    }

    #[test]
    fn test_config_builder() {
        let config = InferenceConfig::new()
            .with_threads(8)
            .with_precision(PrecisionMode::High)
            .with_backend(Backend::Metal);

        assert_eq!(config.thread_count, 8);
        assert_eq!(config.precision_mode, PrecisionMode::High);
        assert_eq!(config.backend, Backend::Metal);
    }
}
