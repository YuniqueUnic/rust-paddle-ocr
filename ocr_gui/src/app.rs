use crate::config::{OcrConfig, Theme};
use crate::data::{HistoryRecord, HistoryTextBlock, TextBlock, WorkMode};
use crate::theme;
use crate::ui::{icon_bar, image_viewer, settings, status_bar, text_editor};
use anyhow::Result;
use egui::TextureHandle;
use image::DynamicImage;
use ocr_rs::engine::{OcrEngine, OcrEngineConfig};
use ocr_rs::mnn::Backend;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

pub struct OcrApp {
    // OCR engine
    engine: Option<OcrEngine>,
    
    // Current work mode
    current_mode: WorkMode,
    
    // Image data
    current_image: Option<DynamicImage>,
    current_image_path: Option<PathBuf>,
    image_texture: Option<TextureHandle>,
    
    // OCR results
    text_blocks: Vec<TextBlock>,
    
    // Processing state
    is_processing: bool,
    ocr_result_receiver: Option<Receiver<Vec<ocr_rs::engine::OcrResult_>>>,
    
    // UI state
    settings_visible: bool,
    
    // Configuration
    config: OcrConfig,
    
    // Batch mode
    batch_files: Vec<PathBuf>,
    batch_current_index: usize,
    
    // History mode
    history_records: Vec<HistoryRecord>,
    history_selected_index: Option<usize>,
    
    // Error message
    error_message: Option<String>,
}

impl OcrApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = OcrConfig::load().unwrap_or_default();
        theme::apply_theme(&cc.egui_ctx, config.theme);
        
        let mut app = Self {
            engine: None,
            current_mode: WorkMode::File,
            current_image: None,
            current_image_path: None,
            image_texture: None,
            text_blocks: Vec::new(),
            is_processing: false,
            ocr_result_receiver: None,
            settings_visible: false,
            config: config.clone(),
            batch_files: Vec::new(),
            batch_current_index: 0,
            history_records: Vec::new(),
            history_selected_index: None,
            error_message: None,
        };
        
        // Initialize OCR engine
        if let Err(e) = app.init_engine() {
            app.error_message = Some(format!("Failed to initialize OCR engine: {}", e));
        }
        
        // Load history
        app.load_history();
        
        app
    }
    
    pub fn init_engine(&mut self) -> Result<()> {
        let backend = match self.config.backend.as_str() {
            "Metal" => Backend::Metal,
            "OpenCL" => Backend::OpenCL,
            "Vulkan" => Backend::Vulkan,
            "CUDA" => Backend::CUDA,
            _ => Backend::CPU,
        };
        
        let engine_config = OcrEngineConfig::new()
            .with_backend(backend)
            .with_threads(4)
            .with_min_result_confidence(self.config.min_confidence);
        
        self.engine = Some(OcrEngine::new(
            &self.config.det_model_path,
            &self.config.rec_model_path,
            &self.config.charset_path,
            Some(engine_config),
        )?);
        
        Ok(())
    }
    
    fn load_image(&mut self, path: PathBuf) {
        match image::open(&path) {
            Ok(img) => {
                self.current_image = Some(img.clone());
                self.current_image_path = Some(path.clone());
                self.image_texture = None;
                self.text_blocks.clear();
                self.error_message = None;
                
                // Start OCR processing
                self.process_image_async(img);
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load image: {}", e));
            }
        }
    }
    
    fn process_image_async(&mut self, image: DynamicImage) {
        if self.engine.is_none() {
            self.error_message = Some("OCR engine not initialized".to_string());
            return;
        }
        
        self.is_processing = true;
        let (tx, rx) = channel();
        self.ocr_result_receiver = Some(rx);
        
        let engine = self.engine.as_ref().unwrap().clone();
        
        std::thread::spawn(move || {
            let results = engine.recognize(&image);
            let _ = tx.send(results);
        });
    }
    
    fn check_ocr_results(&mut self) {
        if let Some(rx) = &self.ocr_result_receiver {
            if let Ok(results) = rx.try_recv() {
                self.text_blocks = results
                    .into_iter()
                    .map(|r| TextBlock::new(r.text, r.confidence, r.bbox))
                    .collect();
                
                self.is_processing = false;
                self.ocr_result_receiver = None;
                
                // Save to history
                self.save_to_history();
            }
        }
    }
    
    pub fn open_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "gif"])
            .pick_file()
        {
            self.load_image(path);
        }
    }
    
    pub fn open_batch_dialog(&mut self) {
        if let Some(paths) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "gif"])
            .pick_files()
        {
            self.batch_files = paths;
            self.batch_current_index = 0;
            
            if !self.batch_files.is_empty() {
                let path = self.batch_files[0].clone();
                self.load_image(path);
            }
        }
    }
    
    pub fn process_next_batch(&mut self) {
        if self.batch_current_index + 1 < self.batch_files.len() {
            self.batch_current_index += 1;
            let path = self.batch_files[self.batch_current_index].clone();
            self.load_image(path);
        }
    }
    
    fn save_to_history(&mut self) {
        if let Some(image_path) = &self.current_image_path {
            let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
            let results: Vec<HistoryTextBlock> = self.text_blocks.iter().map(|b| b.into()).collect();
            
            let record = HistoryRecord {
                timestamp: timestamp.clone(),
                image_path: image_path.to_string_lossy().to_string(),
                results,
            };
            
            // Save to file
            if let Err(e) = self.save_history_record(&record) {
                tracing::error!("Failed to save history: {}", e);
            }
            
            self.history_records.insert(0, record);
        }
    }
    
    fn save_history_record(&self, record: &HistoryRecord) -> Result<()> {
        let history_dir = PathBuf::from(&self.config.history_dir);
        std::fs::create_dir_all(&history_dir)?;
        
        let record_dir = history_dir.join(&record.timestamp);
        std::fs::create_dir_all(&record_dir)?;
        
        // Save JSON
        let json_path = record_dir.join("result.json");
        let json_content = serde_json::to_string_pretty(record)?;
        std::fs::write(json_path, json_content)?;
        
        // Copy image
        if let Some(image_path) = &self.current_image_path {
            let image_dest = record_dir.join("image.png");
            if let Some(img) = &self.current_image {
                img.save(&image_dest)?;
            }
        }
        
        Ok(())
    }
    
    fn load_history(&mut self) {
        let history_dir = PathBuf::from(&self.config.history_dir);
        if !history_dir.exists() {
            return;
        }
        
        if let Ok(entries) = std::fs::read_dir(&history_dir) {
            for entry in entries.flatten() {
                if let Ok(file_type) = entry.file_type() {
                    if file_type.is_dir() {
                        let json_path = entry.path().join("result.json");
                        if let Ok(content) = std::fs::read_to_string(json_path) {
                            if let Ok(record) = serde_json::from_str::<HistoryRecord>(&content) {
                                self.history_records.push(record);
                            }
                        }
                    }
                }
            }
        }
        
        // Sort by timestamp (newest first)
        self.history_records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    }
    
    pub fn export_text(&self) {
        let text: String = self
            .text_blocks
            .iter()
            .filter(|b| b.selected)
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ocr_result.txt")
            .add_filter("Text", &["txt"])
            .save_file()
        {
            if let Err(e) = std::fs::write(path, text) {
                self.show_error(format!("Failed to export text: {}", e));
            }
        }
    }
    
    pub fn export_json(&self) {
        let selected_blocks: Vec<&TextBlock> = self.text_blocks.iter().filter(|b| b.selected).collect();
        let history_blocks: Vec<HistoryTextBlock> = selected_blocks.iter().map(|b| (*b).into()).collect();
        
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("ocr_result.json")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            if let Ok(json) = serde_json::to_string_pretty(&history_blocks) {
                if let Err(e) = std::fs::write(path, json) {
                    self.show_error(format!("Failed to export JSON: {}", e));
                }
            }
        }
    }
    
    pub fn copy_selected_text(&self) {
        let text: String = self
            .text_blocks
            .iter()
            .filter(|b| b.selected)
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            use cli_clipboard::ClipboardProvider;
            if let Ok(mut clipboard) = cli_clipboard::ClipboardContext::new() {
                let _ = clipboard.set_contents(text);
            }
        }
    }
    
    fn show_error(&self, _msg: String) {
        // In a real implementation, this would show a popup dialog
        tracing::error!("{}", _msg);
    }
    
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            // Ctrl+O: Open file
            if i.modifiers.command && i.key_pressed(egui::Key::O) {
                if self.current_mode == WorkMode::File {
                    self.open_file_dialog();
                } else if self.current_mode == WorkMode::Batch {
                    self.open_batch_dialog();
                }
            }
            
            // Ctrl+A: Select all
            if i.modifiers.command && i.key_pressed(egui::Key::A) {
                for block in &mut self.text_blocks {
                    block.selected = true;
                }
            }
            
            // Ctrl+Shift+A: Deselect all
            if i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::A) {
                for block in &mut self.text_blocks {
                    block.selected = false;
                }
            }
            
            // Ctrl+R: Reload/Reprocess
            if i.modifiers.command && i.key_pressed(egui::Key::R) {
                if let Some(img) = self.current_image.clone() {
                    self.process_image_async(img);
                }
            }
            
            // Ctrl+E: Export text
            if i.modifiers.command && i.key_pressed(egui::Key::E) {
                self.export_text();
            }
        });
    }
    
    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        ctx.input(|i| {
            if !i.raw.dropped_files.is_empty() {
                for file in &i.raw.dropped_files {
                    if let Some(path) = &file.path {
                        self.load_image(path.clone());
                        break;
                    }
                }
            }
        });
    }
}

impl eframe::App for OcrApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply theme
        theme::apply_theme(ctx, self.config.theme);
        
        // Check for OCR results
        self.check_ocr_results();
        
        // Handle shortcuts and drag-drop
        self.handle_shortcuts(ctx);
        self.handle_drag_and_drop(ctx);
        
        // Show error message if any
        if let Some(msg) = &self.error_message {
            egui::Window::new("Error")
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.label(msg);
                    if ui.button("OK").clicked() {
                        self.error_message = None;
                    }
                });
        }
        
        // Show settings window
        if self.settings_visible {
            settings::show(ctx, &mut self.settings_visible, &mut self.config, self);
        }
        
        // Left icon bar
        egui::SidePanel::left("icon_bar")
            .resizable(false)
            .exact_width(60.0)
            .show(ctx, |ui| {
                icon_bar::show(ui, self);
            });
        
        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            // Right text editor panel
            egui::SidePanel::right("text_panel")
                .resizable(true)
                .default_width(400.0)
                .min_width(300.0)
                .show_inside(ui, |ui| {
                    text_editor::show(ui, self);
                });
            
            // Center image viewer
            egui::CentralPanel::default().show_inside(ui, |ui| {
                image_viewer::show(ui, self);
            });
        });
        
        // Bottom status bar
        status_bar::show(ctx, self);
        
        // Request repaint if processing
        if self.is_processing {
            ctx.request_repaint();
        }
    }
    
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        if let Err(e) = self.config.save() {
            tracing::error!("Failed to save config: {}", e);
        }
    }
}
