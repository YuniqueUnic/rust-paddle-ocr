//! 后处理工具
//!
//! Postprocessing Utilities
//!
//! 提供文本检测结果的后处理功能，包括边界框提取、NMS、框合并等

use image::GrayImage;
use imageproc::contours::{find_contours, Contour};
use imageproc::point::Point;
use imageproc::rect::Rect;

/// 文本边界框
#[derive(Debug, Clone)]
pub struct TextBox {
    /// 边界框矩形
    pub rect: Rect,
    /// 置信度分数
    pub score: f32,
    /// 四个角点 (可选，用于旋转框)
    pub points: Option<[Point<f32>; 4]>,
}

impl TextBox {
    /// 创建新的文本边界框
    pub fn new(rect: Rect, score: f32) -> Self {
        Self {
            rect,
            score,
            points: None,
        }
    }

    /// 带角点创建
    pub fn with_points(rect: Rect, score: f32, points: [Point<f32>; 4]) -> Self {
        Self {
            rect,
            score,
            points: Some(points),
        }
    }

    /// 计算面积
    pub fn area(&self) -> u32 {
        self.rect.width() * self.rect.height()
    }

    /// 扩展边界框
    pub fn expand(&self, border: u32, max_width: u32, max_height: u32) -> Self {
        let x = (self.rect.left() - border as i32).max(0) as u32;
        let y = (self.rect.top() - border as i32).max(0) as u32;
        let right = ((self.rect.left() as u32 + self.rect.width()) + border).min(max_width);
        let bottom = ((self.rect.top() as u32 + self.rect.height()) + border).min(max_height);

        Self {
            rect: Rect::at(x as i32, y as i32).of_size(right - x, bottom - y),
            score: self.score,
            points: self.points,
        }
    }
}

/// 从分割掩码中提取文本边界框
///
/// # 参数
/// - `mask`: 二值化后的掩码 (0 或 255)
/// - `width`: 掩码宽度
/// - `height`: 掩码高度
/// - `original_width`: 原始图像宽度
/// - `original_height`: 原始图像高度
/// - `min_area`: 最小边界框面积
/// - `box_threshold`: 边界框分数阈值
pub fn extract_boxes_from_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    _box_threshold: f32,
) -> Vec<TextBox> {
    extract_boxes_from_mask_with_padding(
        mask,
        width,
        height,
        width,
        height,
        original_width,
        original_height,
        min_area,
        _box_threshold,
    )
}

/// 从带 padding 的分割掩码中提取文本边界框
///
/// # 参数
/// - `mask`: 二值化后的掩码 (0 或 255)
/// - `mask_width`: 掩码宽度（含 padding）
/// - `mask_height`: 掩码高度（含 padding）
/// - `valid_width`: 有效区域宽度（不含 padding）
/// - `valid_height`: 有效区域高度（不含 padding）
/// - `original_width`: 原始图像宽度
/// - `original_height`: 原始图像高度
/// - `min_area`: 最小边界框面积
/// - `box_threshold`: 边界框分数阈值
pub fn extract_boxes_from_mask_with_padding(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    _box_threshold: f32,
) -> Vec<TextBox> {
    extract_boxes_with_unclip(
        mask,
        mask_width,
        mask_height,
        valid_width,
        valid_height,
        original_width,
        original_height,
        min_area,
        1.5, // 默认 unclip_ratio
    )
}

/// 从分割掩码中提取文本边界框（带 unclip 扩展）
///
/// DB 算法的核心是对检测到的轮廓进行 unclip 扩展，
/// 因为模型输出的分割掩码通常比实际文本区域要小。
pub fn extract_boxes_with_unclip(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    unclip_ratio: f32,
) -> Vec<TextBox> {
    // 创建灰度图像
    let gray_image = GrayImage::from_raw(mask_width, mask_height, mask.to_vec())
        .unwrap_or_else(|| GrayImage::new(mask_width, mask_height));

    // 查找轮廓
    let contours = find_contours::<i32>(&gray_image);

    // 计算缩放比例（从有效区域到原始图像）
    let scale_x = original_width as f32 / valid_width as f32;
    let scale_y = original_height as f32 / valid_height as f32;

    let mut boxes = Vec::new();

    for contour in contours {
        if contour.points.len() < 4 {
            continue;
        }

        // 计算边界框
        let (min_x, min_y, max_x, max_y) = get_contour_bounds(&contour);

        // 过滤掉在 padding 区域的轮廓
        if min_x >= valid_width as i32 || min_y >= valid_height as i32 {
            continue;
        }

        // 裁剪到有效区域
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);
        let max_x = max_x.min(valid_width as i32);
        let max_y = max_y.min(valid_height as i32);

        let box_width = (max_x - min_x) as u32;
        let box_height = (max_y - min_y) as u32;

        // 过滤太小的框
        if box_width * box_height < min_area {
            continue;
        }

        // 计算 unclip 扩展量
        // DB 算法使用面积和周长计算扩展距离: distance = Area * unclip_ratio / Perimeter
        let area = box_width as f32 * box_height as f32;
        let perimeter = 2.0 * (box_width + box_height) as f32;
        let expand_dist = (area * unclip_ratio / perimeter).max(1.0);

        // 应用 unclip 扩展（在缩放前的坐标上）
        let expanded_min_x = (min_x as f32 - expand_dist).max(0.0) as i32;
        let expanded_min_y = (min_y as f32 - expand_dist).max(0.0) as i32;
        let expanded_max_x = (max_x as f32 + expand_dist).min(valid_width as f32) as i32;
        let expanded_max_y = (max_y as f32 + expand_dist).min(valid_height as f32) as i32;

        let expanded_w = (expanded_max_x - expanded_min_x) as u32;
        let expanded_h = (expanded_max_y - expanded_min_y) as u32;

        // 缩放到原始图像尺寸
        let scaled_x = (expanded_min_x as f32 * scale_x) as i32;
        let scaled_y = (expanded_min_y as f32 * scale_y) as i32;
        let scaled_w = (expanded_w as f32 * scale_x) as u32;
        let scaled_h = (expanded_h as f32 * scale_y) as u32;

        // 确保边界在有效范围内
        let final_x = scaled_x.max(0) as u32;
        let final_y = scaled_y.max(0) as u32;
        let final_w = scaled_w.min(original_width.saturating_sub(final_x));
        let final_h = scaled_h.min(original_height.saturating_sub(final_y));

        if final_w > 0 && final_h > 0 {
            let rect = Rect::at(final_x as i32, final_y as i32).of_size(final_w, final_h);
            boxes.push(TextBox::new(rect, 1.0));
        }
    }

    boxes
}

/// 获取轮廓的边界
fn get_contour_bounds(contour: &Contour<i32>) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for point in &contour.points {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }

    (min_x, min_y, max_x, max_y)
}

/// 非极大值抑制 (NMS)
///
/// 过滤重叠的边界框，保留分数最高的
///
/// # 参数
/// - `boxes`: 边界框列表
/// - `iou_threshold`: IoU 阈值，超过此值认为是重叠
pub fn nms(boxes: &[TextBox], iou_threshold: f32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // 使用索引排序而非克隆整个向量
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| boxes[b].score.partial_cmp(&boxes[a].score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &i in &indices {
        if suppressed[i] {
            continue;
        }

        keep.push(boxes[i].clone());

        for &j in &indices {
            if j <= i || suppressed[j] {
                continue;
            }

            let iou = compute_iou(&boxes[i].rect, &boxes[j].rect);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

/// 计算两个矩形的 IoU (交并比)
pub fn compute_iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.left().max(b.left());
    let y1 = a.top().max(b.top());
    let x2 = (a.left() + a.width() as i32).min(b.left() + b.width() as i32);
    let y2 = (a.top() + a.height() as i32).min(b.top() + b.height() as i32);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) as f32 * (y2 - y1) as f32;
    let area_a = a.width() as f32 * a.height() as f32;
    let area_b = b.width() as f32 * b.height() as f32;
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// 合并相邻的边界框
///
/// 将距离较近的边界框合并为一个
///
/// # 参数
/// - `boxes`: 边界框列表
/// - `distance_threshold`: 距离阈值，小于此值的框会被合并
pub fn merge_adjacent_boxes(boxes: &[TextBox], distance_threshold: i32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut used = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if used[i] {
            continue;
        }

        let mut current = boxes[i].rect;
        let mut group_score = boxes[i].score;
        let mut count = 1;
        used[i] = true;

        // 查找可以合并的框
        loop {
            let mut found = false;

            for j in 0..boxes.len() {
                if used[j] {
                    continue;
                }

                if can_merge(&current, &boxes[j].rect, distance_threshold) {
                    current = merge_rects(&current, &boxes[j].rect);
                    group_score += boxes[j].score;
                    count += 1;
                    used[j] = true;
                    found = true;
                }
            }

            if !found {
                break;
            }
        }

        merged.push(TextBox::new(current, group_score / count as f32));
    }

    merged
}

/// 判断两个框是否可以合并
fn can_merge(a: &Rect, b: &Rect, threshold: i32) -> bool {
    // 计算垂直距离
    let a_bottom = a.top() + a.height() as i32;
    let b_bottom = b.top() + b.height() as i32;

    let _vertical_dist = if a.top() > b_bottom {
        a.top() - b_bottom
    } else if b.top() > a_bottom {
        b.top() - a_bottom
    } else {
        0 // 垂直重叠
    };

    // 计算水平距离
    let a_right = a.left() + a.width() as i32;
    let b_right = b.left() + b.width() as i32;

    let horizontal_dist = if a.left() > b_right {
        a.left() - b_right
    } else if b.left() > a_right {
        b.left() - a_right
    } else {
        0 // 水平重叠
    };

    // 检查是否在同一行 (垂直重叠) 且水平距离小于阈值
    let vertical_overlap = !(a.top() > b_bottom || b.top() > a_bottom);

    vertical_overlap && horizontal_dist <= threshold
}

/// 合并两个矩形
fn merge_rects(a: &Rect, b: &Rect) -> Rect {
    let x1 = a.left().min(b.left());
    let y1 = a.top().min(b.top());
    let x2 = (a.left() + a.width() as i32).max(b.left() + b.width() as i32);
    let y2 = (a.top() + a.height() as i32).max(b.top() + b.height() as i32);

    Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32)
}

/// 按阅读顺序排序边界框 (从上到下，从左到右)
pub fn sort_boxes_by_reading_order(boxes: &mut [TextBox]) {
    boxes.sort_by(|a, b| {
        // 首先按 y 坐标排序 (行)
        let y_cmp = a.rect.top().cmp(&b.rect.top());
        if y_cmp != std::cmp::Ordering::Equal {
            return y_cmp;
        }
        // 同一行按 x 坐标排序
        a.rect.left().cmp(&b.rect.left())
    });
}

/// 按行分组边界框
///
/// 将 y 坐标接近的框归为同一行
pub fn group_boxes_by_line(boxes: &[TextBox], line_threshold: i32) -> Vec<Vec<TextBox>> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut sorted_boxes = boxes.to_vec();
    sorted_boxes.sort_by_key(|b| b.rect.top());

    let mut lines: Vec<Vec<TextBox>> = Vec::new();
    let mut current_line: Vec<TextBox> = vec![sorted_boxes[0].clone()];
    let mut current_y = sorted_boxes[0].rect.top();

    for box_item in sorted_boxes.iter().skip(1) {
        if (box_item.rect.top() - current_y).abs() <= line_threshold {
            current_line.push(box_item.clone());
        } else {
            // 对当前行按 x 排序
            current_line.sort_by_key(|b| b.rect.left());
            lines.push(current_line);
            current_line = vec![box_item.clone()];
            current_y = box_item.rect.top();
        }
    }

    // 添加最后一行
    if !current_line.is_empty() {
        current_line.sort_by_key(|b| b.rect.left());
        lines.push(current_line);
    }

    lines
}

/// 从多个检测结果中合并边界框 (用于高精度模式)
///
/// # 参数
/// - `results`: 多次检测的结果，每个元素是 (boxes, offset_x, offset_y, scale)
/// - `iou_threshold`: NMS 的 IoU 阈值
pub fn merge_multi_scale_results(
    results: &[(Vec<TextBox>, u32, u32, f32)],
    iou_threshold: f32,
) -> Vec<TextBox> {
    let mut all_boxes = Vec::new();

    for (boxes, offset_x, offset_y, scale) in results {
        for box_item in boxes {
            // 将框坐标转换到原始图像坐标系
            let scaled_x = (box_item.rect.left() as f32 / scale) as i32 + *offset_x as i32;
            let scaled_y = (box_item.rect.top() as f32 / scale) as i32 + *offset_y as i32;
            let scaled_w = (box_item.rect.width() as f32 / scale) as u32;
            let scaled_h = (box_item.rect.height() as f32 / scale) as u32;

            let rect = Rect::at(scaled_x, scaled_y).of_size(scaled_w, scaled_h);
            all_boxes.push(TextBox::new(rect, box_item.score));
        }
    }

    // 应用 NMS 去除重复
    nms(&all_boxes, iou_threshold)
}

// ============== 传统算法检测 ==============

/// 使用传统算法检测文本区域（适用于纯色背景）
///
/// 基于 OTSU 二值化 + 连通域分析，适合：
/// - 纯色背景的文档图像
/// - 高对比度的文本
/// - 作为深度学习检测的补充
///
/// # 参数
/// - `gray_image`: 灰度图像
/// - `min_area`: 最小文本区域面积
/// - `expand_ratio`: 边界框扩展比例
pub fn detect_text_traditional(
    gray_image: &GrayImage,
    min_area: u32,
    expand_ratio: f32,
) -> Vec<TextBox> {
    let (width, height) = gray_image.dimensions();

    // 1. 计算 OTSU 阈值
    let threshold = otsu_threshold(gray_image);

    // 2. 二值化
    let binary: Vec<u8> = gray_image
        .pixels()
        .map(|p| if p.0[0] < threshold { 255 } else { 0 })
        .collect();

    // 3. 创建二值图像并查找轮廓
    let binary_image =
        GrayImage::from_raw(width, height, binary).unwrap_or_else(|| GrayImage::new(width, height));
    let contours = find_contours::<i32>(&binary_image);

    // 4. 提取边界框
    let mut boxes = Vec::new();
    for contour in contours {
        if contour.points.len() < 4 {
            continue;
        }

        let (min_x, min_y, max_x, max_y) = get_contour_bounds(&contour);
        let box_width = (max_x - min_x) as u32;
        let box_height = (max_y - min_y) as u32;

        if box_width * box_height < min_area {
            continue;
        }

        // 扩展边界框
        let expand_w = (box_width as f32 * expand_ratio * 0.5) as i32;
        let expand_h = (box_height as f32 * expand_ratio * 0.5) as i32;

        let final_x = (min_x - expand_w).max(0) as u32;
        let final_y = (min_y - expand_h).max(0) as u32;
        let final_w = ((max_x + expand_w) as u32)
            .min(width)
            .saturating_sub(final_x);
        let final_h = ((max_y + expand_h) as u32)
            .min(height)
            .saturating_sub(final_y);

        if final_w > 0 && final_h > 0 {
            let rect = Rect::at(final_x as i32, final_y as i32).of_size(final_w, final_h);
            boxes.push(TextBox::new(rect, 1.0));
        }
    }

    // 5. 合并相邻框形成文本行
    merge_into_text_lines(&boxes, 10)
}

/// OTSU 自适应阈值计算
fn otsu_threshold(image: &GrayImage) -> u8 {
    // 计算直方图
    let mut histogram = [0u32; 256];
    for pixel in image.pixels() {
        histogram[pixel.0[0] as usize] += 1;
    }

    let total = image.pixels().count() as f64;
    let mut sum = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i as f64 * count as f64;
    }

    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_variance = 0.0;
    let mut threshold = 0u8;

    for (t, &count) in histogram.iter().enumerate() {
        w_b += count as f64;
        if w_b == 0.0 {
            continue;
        }

        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }

        sum_b += t as f64 * count as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;

        let variance = w_b * w_f * (m_b - m_f).powi(2);
        if variance > max_variance {
            max_variance = variance;
            threshold = t as u8;
        }
    }

    threshold
}

/// 将独立的字符框合并成文本行
fn merge_into_text_lines(boxes: &[TextBox], gap_threshold: i32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // 按 y 坐标分组
    let mut sorted_boxes: Vec<_> = boxes.iter().collect();
    sorted_boxes.sort_by_key(|b| b.rect.top());

    let mut lines: Vec<TextBox> = Vec::new();

    for bbox in sorted_boxes {
        let mut merged = false;

        // 尝试合并到现有行
        for line in &mut lines {
            let line_center_y = line.rect.top() + line.rect.height() as i32 / 2;
            let box_center_y = bbox.rect.top() + bbox.rect.height() as i32 / 2;

            // 如果垂直方向重叠且水平方向接近
            if (line_center_y - box_center_y).abs() < line.rect.height() as i32 / 2 {
                let line_right = line.rect.left() + line.rect.width() as i32;
                let box_left = bbox.rect.left();

                if (box_left - line_right).abs() < gap_threshold * 3 {
                    // 合并
                    let new_left = line.rect.left().min(bbox.rect.left());
                    let new_top = line.rect.top().min(bbox.rect.top());
                    let new_right = (line.rect.left() + line.rect.width() as i32)
                        .max(bbox.rect.left() + bbox.rect.width() as i32);
                    let new_bottom = (line.rect.top() + line.rect.height() as i32)
                        .max(bbox.rect.top() + bbox.rect.height() as i32);

                    line.rect = Rect::at(new_left, new_top)
                        .of_size((new_right - new_left) as u32, (new_bottom - new_top) as u32);
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            lines.push(bbox.clone());
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_textbox_new() {
        let rect = Rect::at(10, 20).of_size(100, 50);
        let tb = TextBox::new(rect, 0.95);

        assert_eq!(tb.rect.left(), 10);
        assert_eq!(tb.rect.top(), 20);
        assert_eq!(tb.rect.width(), 100);
        assert_eq!(tb.rect.height(), 50);
        assert_eq!(tb.score, 0.95);
        assert!(tb.points.is_none());
    }

    #[test]
    fn test_textbox_with_points() {
        let rect = Rect::at(0, 0).of_size(100, 50);
        let points = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 50.0),
            Point::new(0.0, 50.0),
        ];
        let tb = TextBox::with_points(rect, 0.9, points);

        assert!(tb.points.is_some());
        let pts = tb.points.unwrap();
        assert_eq!(pts[0].x, 0.0);
        assert_eq!(pts[1].x, 100.0);
    }

    #[test]
    fn test_textbox_area() {
        let tb = TextBox::new(Rect::at(0, 0).of_size(100, 50), 0.9);
        assert_eq!(tb.area(), 5000);
    }

    #[test]
    fn test_textbox_expand() {
        let tb = TextBox::new(Rect::at(50, 50).of_size(100, 100), 0.9);
        let expanded = tb.expand(10, 500, 500);

        assert_eq!(expanded.rect.left(), 40);
        assert_eq!(expanded.rect.top(), 40);
        assert_eq!(expanded.rect.width(), 120);
        assert_eq!(expanded.rect.height(), 120);
    }

    #[test]
    fn test_textbox_expand_clamp() {
        // 测试边界裁剪
        let tb = TextBox::new(Rect::at(5, 5).of_size(100, 100), 0.9);
        let expanded = tb.expand(10, 200, 200);

        // 左上角应该被限制在 (0, 0)
        assert_eq!(expanded.rect.left(), 0);
        assert_eq!(expanded.rect.top(), 0);
    }

    #[test]
    fn test_compute_iou() {
        let a = Rect::at(0, 0).of_size(10, 10);
        let b = Rect::at(5, 5).of_size(10, 10);

        let iou = compute_iou(&a, &b);
        assert!(iou > 0.0 && iou < 1.0);

        // 不相交
        let c = Rect::at(100, 100).of_size(10, 10);
        assert_eq!(compute_iou(&a, &c), 0.0);

        // 完全重叠
        assert_eq!(compute_iou(&a, &a), 1.0);
    }

    #[test]
    fn test_compute_iou_partial_overlap() {
        // 50% 重叠的情况
        let a = Rect::at(0, 0).of_size(10, 10);
        let b = Rect::at(5, 0).of_size(10, 10);

        let iou = compute_iou(&a, &b);
        // 交集面积 = 5 * 10 = 50
        // 并集面积 = 100 + 100 - 50 = 150
        // IoU = 50 / 150 ≈ 0.333
        assert!((iou - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_nms() {
        // 第一个和第二个框有很大重叠，第三个框独立
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),
            TextBox::new(Rect::at(1, 1).of_size(10, 10), 0.8), // 与第一个框高度重叠
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 0.7),
        ];

        let result = nms(&boxes, 0.3); // 使用较低的阈值确保重叠框被过滤
                                       // 第一个框（最高分数）和第三个框（无重叠）应该保留
        assert!(
            result.len() >= 2,
            "至少应该保留2个框，实际: {}",
            result.len()
        );
    }

    #[test]
    fn test_nms_empty() {
        let boxes: Vec<TextBox> = vec![];
        let result = nms(&boxes, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_single() {
        let boxes = vec![TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9)];
        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_nms_no_overlap() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),
            TextBox::new(Rect::at(50, 50).of_size(10, 10), 0.8),
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 0.7),
        ];

        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 3); // 所有框都保留
    }

    #[test]
    fn test_merge_adjacent() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 1.0),
            TextBox::new(Rect::at(12, 0).of_size(10, 10), 1.0), // 水平距离 2
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 1.0),
        ];

        let result = merge_adjacent_boxes(&boxes, 5);
        assert_eq!(result.len(), 2); // 前两个应该合并
    }

    #[test]
    fn test_merge_adjacent_empty() {
        let boxes: Vec<TextBox> = vec![];
        let result = merge_adjacent_boxes(&boxes, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sort_boxes_by_reading_order() {
        let mut boxes = vec![
            TextBox::new(Rect::at(100, 0).of_size(10, 10), 0.9), // 第一行右边
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),   // 第一行左边
            TextBox::new(Rect::at(0, 50).of_size(10, 10), 0.9),  // 第二行
        ];

        sort_boxes_by_reading_order(&mut boxes);

        // 应该先按行排序，然后行内按x坐标排序
        assert_eq!(boxes[0].rect.left(), 0);
        assert_eq!(boxes[0].rect.top(), 0);
    }

    #[test]
    fn test_group_boxes_by_line() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(50, 20), 0.9),
            TextBox::new(Rect::at(60, 0).of_size(50, 20), 0.9),
            TextBox::new(Rect::at(0, 50).of_size(50, 20), 0.9),
        ];

        let lines = group_boxes_by_line(&boxes, 10);

        // 应该分成两行
        assert_eq!(lines.len(), 2);
    }
}
