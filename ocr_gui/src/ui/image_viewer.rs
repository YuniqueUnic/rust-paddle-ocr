use crate::app::OcrApp;
use crate::data::WorkMode;

pub fn show(ui: &mut egui::Ui, app: &mut OcrApp) {
    match app.current_mode {
        WorkMode::File | WorkMode::Batch => show_image_view(ui, app),
        WorkMode::History => show_history_view(ui, app),
    }
}

fn show_image_view(ui: &mut egui::Ui, app: &mut OcrApp) {
    if app.current_image.is_none() {
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.heading("No Image Loaded");
                ui.add_space(10.0);
                ui.label("Drag and drop an image here");
                ui.label("or");
                if ui.button("Open Image").clicked() {
                    app.open_file_dialog();
                }
            });
        });
        return;
    }
    
    // Load texture if needed
    if app.image_texture.is_none() {
        if let Some(img) = &app.current_image {
            let size = [img.width() as usize, img.height() as usize];
            let image_buffer = img.to_rgba8();
            let pixels = image_buffer.as_flat_samples();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
            app.image_texture = Some(ui.ctx().load_texture(
                "current_image",
                color_image,
                Default::default(),
            ));
        }
    }
    
    if let Some(texture) = &app.image_texture {
        let available_size = ui.available_size();
        let image_size = texture.size_vec2();
        
        // Calculate scale to fit
        let scale = (available_size.x / image_size.x)
            .min(available_size.y / image_size.y)
            .min(1.0);
        
        let scaled_size = image_size * scale;
        
        // Center the image
        let offset = (available_size - scaled_size) * 0.5;
        
        let image_rect = egui::Rect::from_min_size(
            ui.min_rect().min + offset.to_vec2(),
            scaled_size,
        );
        
        // Draw image
        let response = ui.allocate_rect(image_rect, egui::Sense::click());
        ui.painter().image(
            texture.id(),
            image_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        
        // Draw text boxes
        draw_text_boxes(ui, app, image_rect, image_size);
        
        // Handle clicks
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                handle_box_click(app, pos, image_rect, image_size);
            }
        }
    }
}

fn draw_text_boxes(ui: &mut egui::Ui, app: &OcrApp, image_rect: egui::Rect, image_size: egui::Vec2) {
    let painter = ui.painter();
    let scale = image_rect.size() / image_size;
    
    for (idx, block) in app.text_blocks.iter().enumerate() {
        let color = if block.selected {
            egui::Color32::from_rgba_unmultiplied(0, 255, 0, 200)
        } else {
            egui::Color32::from_rgba_unmultiplied(128, 128, 128, 150)
        };
        
        // Convert bbox points to screen coordinates
        let points: Vec<egui::Pos2> = block
            .bbox
            .points
            .iter()
            .map(|p| {
                egui::pos2(
                    image_rect.min.x + p[0] * scale.x,
                    image_rect.min.y + p[1] * scale.y,
                )
            })
            .collect();
        
        if points.len() == 4 {
            // Draw polygon
            painter.add(egui::Shape::closed_line(
                points.clone(),
                egui::Stroke::new(2.0, color),
            ));
            
            // Fill with transparency
            let fill_color = if block.selected {
                egui::Color32::from_rgba_unmultiplied(0, 255, 0, 30)
            } else {
                egui::Color32::from_rgba_unmultiplied(128, 128, 128, 20)
            };
            painter.add(egui::Shape::convex_polygon(
                points.clone(),
                fill_color,
                egui::Stroke::NONE,
            ));
            
            // Draw label
            if block.selected {
                let label_text = format!("#{} ({:.0}%)", idx + 1, block.confidence * 100.0);
                painter.text(
                    points[0],
                    egui::Align2::LEFT_TOP,
                    label_text,
                    egui::FontId::proportional(14.0),
                    egui::Color32::WHITE,
                );
            }
        }
    }
}

fn handle_box_click(app: &mut OcrApp, pos: egui::Pos2, image_rect: egui::Rect, image_size: egui::Vec2) {
    let scale = image_rect.size() / image_size;
    
    // Convert click position to image coordinates
    let image_x = (pos.x - image_rect.min.x) / scale.x;
    let image_y = (pos.y - image_rect.min.y) / scale.y;
    
    // Find clicked box (iterate in reverse to prioritize top boxes)
    for block in app.text_blocks.iter_mut().rev() {
        if point_in_polygon(image_x, image_y, &block.bbox.points) {
            block.selected = !block.selected;
            return;
        }
    }
    
    // If clicked outside any box, deselect all
    for block in app.text_blocks.iter_mut() {
        block.selected = false;
    }
}

fn point_in_polygon(x: f32, y: f32, points: &[[f32; 2]]) -> bool {
    if points.len() < 3 {
        return false;
    }
    
    let mut inside = false;
    let mut j = points.len() - 1;
    
    for i in 0..points.len() {
        let xi = points[i][0];
        let yi = points[i][1];
        let xj = points[j][0];
        let yj = points[j][1];
        
        let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    
    inside
}

fn show_history_view(ui: &mut egui::Ui, app: &mut OcrApp) {
    ui.vertical(|ui| {
        ui.heading("History Records");
        ui.separator();
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (idx, record) in app.history_records.iter().enumerate() {
                let is_selected = app.history_selected_index == Some(idx);
                
                let response = ui.selectable_label(is_selected, format!("📅 {}", record.timestamp));
                
                if response.clicked() {
                    app.history_selected_index = Some(idx);
                    
                    // Load the history image
                    let history_dir = std::path::PathBuf::from(&app.config.history_dir);
                    let image_path = history_dir.join(&record.timestamp).join("image.png");
                    
                    if image_path.exists() {
                        if let Ok(img) = image::open(&image_path) {
                            app.current_image = Some(img);
                            app.image_texture = None;
                            
                            // Convert history blocks back to text blocks
                            app.text_blocks = record
                                .results
                                .iter()
                                .map(|hb| {
                                    crate::data::TextBlock {
                                        text: hb.text.clone(),
                                        confidence: hb.confidence,
                                        bbox: ocr_rs::postprocess::TextBox {
                                            points: hb.bbox.clone().try_into().unwrap_or([[0.0, 0.0]; 4]),
                                        },
                                        selected: false,
                                    }
                                })
                                .collect();
                        }
                    }
                }
                
                if is_selected {
                    ui.indent(idx, |ui| {
                        ui.label(format!("📁 {}", record.image_path));
                        ui.label(format!("📝 {} text blocks", record.results.len()));
                    });
                }
            }
            
            if app.history_records.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label("No history records");
                });
            }
        });
    });
}
