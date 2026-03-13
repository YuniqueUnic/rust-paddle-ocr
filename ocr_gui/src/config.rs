use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Theme {
    Dark,
    Light,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrConfig {
    pub det_model_path: String,
    pub rec_model_path: String,
    pub charset_path: String,
    pub theme: Theme,
    pub backend: String,
    pub min_confidence: f32,
    pub history_dir: String,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            det_model_path: "models/PP-OCRv5_mobile_det.mnn".to_string(),
            rec_model_path: "models/PP-OCRv5_mobile_rec.mnn".to_string(),
            charset_path: "models/ppocr_keys_v5.txt".to_string(),
            theme: Theme::Dark,
            backend: if cfg!(target_os = "macos") {
                "Metal".to_string()
            } else if cfg!(target_os = "windows") {
                "Vulkan".to_string()
            } else {
                "OpenCL".to_string()
            },
            min_confidence: 0.7,
            history_dir: "ocr_history".to_string(),
        }
    }
}

impl OcrConfig {
    pub fn load() -> anyhow::Result<Self> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = std::fs::read_to_string(config_path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self::default())
        }
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(config_path, content)?;

        Ok(())
    }

    fn config_path() -> anyhow::Result<PathBuf> {
        let config_dir =
            dirs::config_dir().ok_or_else(|| anyhow::anyhow!("Unable to get config directory"))?;

        Ok(config_dir.join("ocr_gui").join("config.json"))
    }
}
