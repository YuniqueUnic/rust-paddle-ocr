//! 文本检测模型
//!
//! Text Detection Model
//!
//! 提供基于 PaddleOCR 检测模型的文本区域检测功能

use image::{DynamicImage, GenericImageView};
use ndarray::ArrayD;
use std::path::Path;

use crate::error::{OcrError, OcrResult};
use crate::mnn::{InferenceConfig, InferenceEngine};
use crate::postprocess::{extract_boxes_with_unclip, TextBox};
use crate::preprocess::{preprocess_for_det, NormalizeParams};

/// 检测精度模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetPrecisionMode {
    /// 快速模式 - 单次检测
    #[default]
    Fast,
}

/// 检测选项
#[derive(Debug, Clone)]
pub struct DetOptions {
    /// 图片最大边长限制 (超过会缩放)
    pub max_side_len: u32,
    /// 边界框二值化阈值 (0.0 - 1.0)
    pub box_threshold: f32,
    /// 文本框扩展比例
    pub unclip_ratio: f32,
    /// 像素级分割阈值
    pub score_threshold: f32,
    /// 最小边界框面积
    pub min_area: u32,
    /// 边界框边距扩展
    pub box_border: u32,
    /// 是否合并相邻文本框
    pub merge_boxes: bool,
    /// 合并距离阈值
    pub merge_threshold: i32,
    /// 精度模式
    pub precision_mode: DetPrecisionMode,
    /// 多尺度检测的缩放比例列表 (仅高精度模式)
    pub multi_scales: Vec<f32>,
    /// 分块检测的块大小 (仅高精度模式)
    pub block_size: u32,
    /// 分块检测的重叠区域
    pub block_overlap: u32,
    /// NMS IoU 阈值
    pub nms_threshold: f32,
}

impl Default for DetOptions {
    fn default() -> Self {
        Self {
            max_side_len: 960,
            box_threshold: 0.5,
            unclip_ratio: 1.5,
            score_threshold: 0.3,
            min_area: 16,
            box_border: 5,
            merge_boxes: false,
            merge_threshold: 10,
            precision_mode: DetPrecisionMode::Fast,
            multi_scales: vec![0.5, 1.0, 1.5],
            block_size: 640,
            block_overlap: 100,
            nms_threshold: 0.3,
        }
    }
}

impl DetOptions {
    /// 创建新的检测选项
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置最大边长
    pub fn with_max_side_len(mut self, len: u32) -> Self {
        self.max_side_len = len;
        self
    }

    /// 设置边界框阈值
    pub fn with_box_threshold(mut self, threshold: f32) -> Self {
        self.box_threshold = threshold;
        self
    }

    /// 设置分割阈值
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// 设置最小面积
    pub fn with_min_area(mut self, area: u32) -> Self {
        self.min_area = area;
        self
    }

    /// 设置边框扩展
    pub fn with_box_border(mut self, border: u32) -> Self {
        self.box_border = border;
        self
    }

    /// 启用框合并
    pub fn with_merge_boxes(mut self, merge: bool) -> Self {
        self.merge_boxes = merge;
        self
    }

    /// 设置合并阈值
    pub fn with_merge_threshold(mut self, threshold: i32) -> Self {
        self.merge_threshold = threshold;
        self
    }

    /// 设置精度模式
    pub fn with_precision_mode(mut self, mode: DetPrecisionMode) -> Self {
        self.precision_mode = mode;
        self
    }

    /// 设置多尺度比例
    pub fn with_multi_scales(mut self, scales: Vec<f32>) -> Self {
        self.multi_scales = scales;
        self
    }

    /// 设置分块大小
    pub fn with_block_size(mut self, size: u32) -> Self {
        self.block_size = size;
        self
    }

    /// 快速模式预设
    pub fn fast() -> Self {
        Self {
            max_side_len: 960,
            precision_mode: DetPrecisionMode::Fast,
            ..Default::default()
        }
    }
}

/// 文本检测模型
pub struct DetModel {
    engine: InferenceEngine,
    options: DetOptions,
    normalize_params: NormalizeParams,
}

impl DetModel {
    /// 从模型文件创建检测器
    ///
    /// # 参数
    /// - `model_path`: 模型文件路径 (.mnn 格式)
    /// - `config`: 可选的推理配置
    pub fn from_file(
        model_path: impl AsRef<Path>,
        config: Option<InferenceConfig>,
    ) -> OcrResult<Self> {
        let engine = InferenceEngine::from_file(model_path, config)?;
        Ok(Self {
            engine,
            options: DetOptions::default(),
            normalize_params: NormalizeParams::paddle_det(),
        })
    }

    /// 从模型字节创建检测器
    pub fn from_bytes(model_bytes: &[u8], config: Option<InferenceConfig>) -> OcrResult<Self> {
        let engine = InferenceEngine::from_buffer(model_bytes, config)?;
        Ok(Self {
            engine,
            options: DetOptions::default(),
            normalize_params: NormalizeParams::paddle_det(),
        })
    }

    /// 设置检测选项
    pub fn with_options(mut self, options: DetOptions) -> Self {
        self.options = options;
        self
    }

    /// 获取当前检测选项
    pub fn options(&self) -> &DetOptions {
        &self.options
    }

    /// 修改检测选项
    pub fn options_mut(&mut self) -> &mut DetOptions {
        &mut self.options
    }

    /// 检测图像中的文本区域
    ///
    /// # 参数
    /// - `image`: 输入图像
    ///
    /// # 返回
    /// 检测到的文本边界框列表
    pub fn detect(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        self.detect_fast(image)
    }

    /// 检测并返回裁剪后的文本图像
    ///
    /// # 参数
    /// - `image`: 输入图像
    ///
    /// # 返回
    /// (文本图像, 对应的边界框) 列表
    pub fn detect_and_crop(&self, image: &DynamicImage) -> OcrResult<Vec<(DynamicImage, TextBox)>> {
        let boxes = self.detect(image)?;
        let (width, height) = image.dimensions();

        let mut results = Vec::with_capacity(boxes.len());

        for text_box in boxes {
            // 扩展边界框
            let expanded = text_box.expand(self.options.box_border, width, height);

            // 裁剪图像
            let cropped = image.crop_imm(
                expanded.rect.left() as u32,
                expanded.rect.top() as u32,
                expanded.rect.width(),
                expanded.rect.height(),
            );

            results.push((cropped, expanded));
        }

        Ok(results)
    }

    /// 快速检测 (单次推理)
    fn detect_fast(&self, image: &DynamicImage) -> OcrResult<Vec<TextBox>> {
        let (original_width, original_height) = image.dimensions();

        // 缩放图像
        let scaled = self.scale_image(image);
        let (scaled_width, scaled_height) = scaled.dimensions();

        // 预处理
        let input = preprocess_for_det(&scaled, &self.normalize_params);

        // 推理 (使用动态形状)
        let output = self.engine.run_dynamic(input.view().into_dyn())?;

        // 后处理 - 输出形状与输入相同（包括 padding）
        let output_shape = output.shape();
        let out_w = output_shape[3] as u32;
        let out_h = output_shape[2] as u32;

        let boxes = self.postprocess_output(
            &output,
            out_w,
            out_h,
            scaled_width,
            scaled_height,
            original_width,
            original_height,
        )?;

        Ok(boxes)
    }

    /// 平衡模式检测 (多尺度)
    /// 缩放图像到最大边长限制
    fn scale_image(&self, image: &DynamicImage) -> DynamicImage {
        let (w, h) = image.dimensions();
        let max_dim = w.max(h);

        if max_dim <= self.options.max_side_len {
            return image.clone();
        }

        let scale = self.options.max_side_len as f64 / max_dim as f64;
        let new_w = (w as f64 * scale).round() as u32;
        let new_h = (h as f64 * scale).round() as u32;

        image.resize_exact(new_w, new_h, image::imageops::FilterType::Lanczos3)
    }

    /// 后处理推理输出
    fn postprocess_output(
        &self,
        output: &ArrayD<f32>,
        out_w: u32,
        out_h: u32,
        scaled_width: u32,
        scaled_height: u32,
        original_width: u32,
        original_height: u32,
    ) -> OcrResult<Vec<TextBox>> {
        // 获取输出数据
        let output_shape = output.shape();
        if output_shape.len() < 3 {
            return Err(OcrError::PostprocessError(
                "检测模型输出形状无效".to_string(),
            ));
        }

        // 提取分割掩码（只取有效区域，去掉 padding）
        let mask_data: Vec<f32> = output.iter().cloned().collect();

        // 二值化
        let binary_mask: Vec<u8> = mask_data
            .iter()
            .map(|&v| {
                if v > self.options.score_threshold {
                    255u8
                } else {
                    0u8
                }
            })
            .collect();

        // 提取边界框（使用 unclip 扩展）
        // DB 算法需要对检测到的轮廓进行扩展，因为模型输出的分割掩码比实际文本区域小
        let boxes = extract_boxes_with_unclip(
            &binary_mask,
            out_w,
            out_h,
            scaled_width,
            scaled_height,
            original_width,
            original_height,
            self.options.min_area,
            self.options.unclip_ratio,
        );

        Ok(boxes)
    }
}

/// 底层检测 API
impl DetModel {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_options_default() {
        let opts = DetOptions::default();
        assert_eq!(opts.max_side_len, 960);
        assert_eq!(opts.box_threshold, 0.5);
        assert_eq!(opts.unclip_ratio, 1.5);
        assert_eq!(opts.score_threshold, 0.3);
        assert_eq!(opts.min_area, 16);
        assert_eq!(opts.box_border, 5);
        assert!(!opts.merge_boxes);
        assert_eq!(opts.merge_threshold, 10);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
        assert_eq!(opts.nms_threshold, 0.3);
    }

    #[test]
    fn test_det_options_fast() {
        let opts = DetOptions::fast();
        assert_eq!(opts.max_side_len, 960);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_options_builder() {
        let opts = DetOptions::new()
            .with_max_side_len(1280)
            .with_box_threshold(0.6)
            .with_score_threshold(0.4)
            .with_min_area(32)
            .with_box_border(10)
            .with_merge_boxes(true)
            .with_merge_threshold(20)
            .with_precision_mode(DetPrecisionMode::Fast)
            .with_multi_scales(vec![0.5, 1.0, 1.5])
            .with_block_size(800);

        assert_eq!(opts.max_side_len, 1280);
        assert_eq!(opts.box_threshold, 0.6);
        assert_eq!(opts.score_threshold, 0.4);
        assert_eq!(opts.min_area, 32);
        assert_eq!(opts.box_border, 10);
        assert!(opts.merge_boxes);
        assert_eq!(opts.merge_threshold, 20);
        assert_eq!(opts.precision_mode, DetPrecisionMode::Fast);
        assert_eq!(opts.multi_scales, vec![0.5, 1.0, 1.5]);
        assert_eq!(opts.block_size, 800);
    }

    #[test]
    fn test_det_precision_mode_default() {
        let mode = DetPrecisionMode::default();
        assert_eq!(mode, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_precision_mode_equality() {
        assert_eq!(DetPrecisionMode::Fast, DetPrecisionMode::Fast);
    }

    #[test]
    fn test_det_options_chaining() {
        // 测试链式调用不会丢失之前的设置
        let opts = DetOptions::new()
            .with_max_side_len(1000)
            .with_box_threshold(0.7);

        assert_eq!(opts.max_side_len, 1000);
        assert_eq!(opts.box_threshold, 0.7);
        // 其他值应该是默认值
        assert_eq!(opts.score_threshold, 0.3);
    }

    #[test]
    fn test_det_options_presets_are_valid() {
        // 确保预设的参数值在有效范围内
        let fast = DetOptions::fast();
        assert!(fast.box_threshold >= 0.0 && fast.box_threshold <= 1.0);
        assert!(fast.score_threshold >= 0.0 && fast.score_threshold <= 1.0);
        assert!(fast.nms_threshold >= 0.0 && fast.nms_threshold <= 1.0);
    }
}
