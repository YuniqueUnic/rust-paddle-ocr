use ocr_rs::postprocess::TextBox;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct TextBlock {
    pub text: String,
    pub confidence: f32,
    pub bbox: TextBox,
    pub selected: bool,
}

impl TextBlock {
    pub fn new(text: String, confidence: f32, bbox: TextBox) -> Self {
        Self {
            text,
            confidence,
            bbox,
            selected: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkMode {
    File,
    Batch,
    History,
}

impl WorkMode {
    pub fn icon(&self) -> &'static str {
        match self {
            WorkMode::File => "📄",
            WorkMode::Batch => "📁",
            WorkMode::History => "📋",
        }
    }
    
    pub fn tooltip(&self) -> &'static str {
        match self {
            WorkMode::File => "Single file OCR",
            WorkMode::Batch => "Batch processing",
            WorkMode::History => "History records",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryRecord {
    pub timestamp: String,
    pub image_path: String,
    pub results: Vec<HistoryTextBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryTextBlock {
    pub text: String,
    pub confidence: f32,
    pub bbox: Vec<[f32; 2]>,
}

impl From<&TextBlock> for HistoryTextBlock {
    fn from(block: &TextBlock) -> Self {
        Self {
            text: block.text.clone(),
            confidence: block.confidence,
            bbox: block.bbox.points.to_vec(),
        }
    }
}
