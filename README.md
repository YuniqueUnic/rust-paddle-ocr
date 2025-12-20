# Rust PaddleOCR

[English](README.md) | [中文](docs/README.zh.md) | [日本語](docs/README.ja.md) | [한국어](docs/README.ko.md)

A lightweight and efficient OCR (Optical Character Recognition) Rust library based on PaddleOCR models. This library leverages the MNN inference framework to provide high-performance text detection and recognition capabilities.

**This is a pure Rust library** focused on providing core OCR functionality. For command-line tools or HTTP services, please refer to:
- 🖥️ **Command-line Tool**: [newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)
- 🌐 **HTTP Service**: [newbee-ocr-service](https://github.com/zibo-chen/newbee-ocr-service)

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## Features

- **Text Detection**: Accurately locate text regions in images
- **Text Recognition**: Recognize text content in detected regions
- **Multiple Model Versions**: Support for PP-OCRv4 and PP-OCRv5 models with flexible selection
- **Multi-language Support**: PP-OCRv5 provides 11+ specialized language models covering 100+ languages
- **Complex Scene Recognition**: Enhanced capabilities for handwritten text, vertical text, and rare character recognition
- **High Performance**: Optimized with MNN inference framework
- **Minimal Dependencies**: Lightweight and easy to integrate
- **Customizable**: Adjustable parameters for different use cases
- **Memory Safe**: Automatic memory management to prevent leaks
- **Pure Rust Implementation**: No external runtime required, cross-platform compatible

## Model Versions

This library supports three PaddleOCR model versions:

### PP-OCRv4
- **Stable Version**: Well-tested with good compatibility
- **Use Cases**: Regular document recognition, scenarios requiring high accuracy
- **Model Files**:
  - Detection model: `ch_PP-OCRv4_det_infer.mnn`
  - Recognition model: `ch_PP-OCRv4_rec_infer.mnn`
  - Character set: `ppocr_keys_v4.txt`

### PP-OCRv5
- **Latest Version**: Next-generation text recognition solution
- **Multi-language Support**: Default model (`PP-OCRv5_mobile_rec.mnn`) supports Simplified Chinese, Traditional Chinese, English, Japanese, and Chinese Pinyin
- **Specialized Language Models**: Provides 11+ specialized models covering 100+ languages for optimal performance
- **Shared Detection Model**: All V5 language models use the same detection model (`PP-OCRv5_mobile_det.mnn`)
- **Enhanced Scene Recognition**:
  - Significantly improved Chinese-English complex handwriting recognition
  - Optimized vertical text recognition
  - Enhanced rare character recognition
- **Performance Improvement**: 13% end-to-end improvement over PP-OCRv4
- **Model Files** (Default multilingual):
  - Detection model: `PP-OCRv5_mobile_det.mnn` (shared across all languages)
  - Recognition model: `PP-OCRv5_mobile_rec.mnn` (default, supports Chinese/English/Japanese)
  - Character set: `ppocr_keys_v5.txt`
- **Specialized Language Model Files** (Optional):
  - Recognition model: `{lang}_PP-OCRv5_mobile_rec_infer.mnn`
  - Character set: `ppocr_keys_{lang}.txt`
  - Available language codes: `arabic`, `cyrillic`, `devanagari`, `el`, `en`, `eslav`, `korean`, `latin`, `ta`, `te`, `th`

#### PP-OCRv5 Language Model Support Details

| Model Name | Supported Languages |
|-----------|-------------------|
| **korean_PP-OCRv5_mobile_rec** | Korean, English |
| **latin_PP-OCRv5_mobile_rec** | French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua |
| **eslav_PP-OCRv5_mobile_rec** | Russian, Belarusian, Ukrainian, English |
| **th_PP-OCRv5_mobile_rec** | Thai, English |
| **el_PP-OCRv5_mobile_rec** | Greek, English |
| **en_PP-OCRv5_mobile_rec** | English |
| **cyrillic_PP-OCRv5_mobile_rec** | Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English |
| **arabic_PP-OCRv5_mobile_rec** | Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English |
| **devanagari_PP-OCRv5_mobile_rec** | Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English |
| **ta_PP-OCRv5_mobile_rec** | Tamil, English |
| **te_PP-OCRv5_mobile_rec** | Telugu, English |

### PP-OCRv5 FP16
- **Efficient Version**: Provides faster inference speed and lower memory usage without sacrificing accuracy
- **Use Cases**: Scenarios requiring high performance and low memory usage
- **Performance Improvements**:
  - Inference speed increased by ~9% (even higher on devices with FP16 acceleration)
  - Memory usage reduced by ~8%
  - Model size reduced by half
- **Model Files**:
  - Detection model: `PP-OCRv5_mobile_det_fp16.mnn`
  - Recognition model: `PP-OCRv5_mobile_rec_fp16.mnn`
  - Character set: `ppocr_keys_v5.txt`

### Model Performance Comparison

| Feature | PP-OCRv4 | PP-OCRv5 | PP-OCRv5 FP16 |
|---------|----------|----------|---------------|
| Language Support | Chinese, English | Multi-language (default supports Chinese/English/Japanese, 11+ specialized models) | Multi-language (default supports Chinese/English/Japanese, 11+ specialized models) |
| Text Type Support | Chinese, English | Simplified Chinese, Traditional Chinese, English, Japanese, Chinese Pinyin | Simplified Chinese, Traditional Chinese, English, Japanese, Chinese Pinyin |
| Handwriting Recognition | Basic | Significantly Enhanced | Significantly Enhanced |
| Vertical Text | Basic | Optimized | Optimized |
| Rare Character Recognition | Limited | Enhanced | Enhanced |
| Inference Speed (FPS) | 1.1 | 1.2 | 1.2 |
| Memory Usage (Peak) | 422.22MB | 388.41MB | 388.41MB |
| Model Size | Standard | Standard | Half |
| Recommended Scenarios | Regular Documents | Complex Scenes & Multilingual | High Performance & Multilingual |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies.rust-paddle-ocr]
git = "https://github.com/zibo-chen/rust-paddle-ocr.git"
```

You can also specify a particular branch or tag:

```toml
[dependencies.rust-paddle-ocr]
git = "https://github.com/zibo-chen/rust-paddle-ocr.git"
branch = "main"
```

### Prerequisites

This library requires:
- Pre-trained PaddleOCR models converted to MNN format
- Character set files for text recognition

## Usage Examples

### Basic Usage

```rust
use rust_paddle_ocr::{OcrEngine, OcrConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create OCR engine configuration
    let config = OcrConfig::default()
        .with_det_model("models/PP-OCRv5_mobile_det.mnn")
        .with_rec_model("models/PP-OCRv5_mobile_rec.mnn")
        .with_keys_file("models/ppocr_keys_v5.txt");
    
    // Initialize OCR engine
    let engine = OcrEngine::new(config)?;
    
    // Perform OCR recognition
    let results = engine.recognize("image.jpg")?;
    
    // Print results
    for result in results {
        println!("Text: {}, Confidence: {:.2}", result.text, result.confidence);
    }
    
    Ok(())
}
```

### Using Specific Language Models

```rust
use rust_paddle_ocr::{OcrEngine, OcrConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use Korean model
    let config = OcrConfig::default()
        .with_det_model("models/PP-OCRv5_mobile_det.mnn")
        .with_rec_model("models/korean_PP-OCRv5_mobile_rec_infer.mnn")
        .with_keys_file("models/ppocr_keys_korean.txt");
    
    let engine = OcrEngine::new(config)?;
    let results = engine.recognize("korean_text.jpg")?;
    
    for result in results {
        println!("{}", result.text);
    }
    
    Ok(())
}
```

For more examples, please refer to the [examples](examples) directory.

## Related Projects

- 🖥️ **[newbee-ocr-cli](https://github.com/zibo-chen/newbee-ocr-cli)** - Command-line tool based on this library, providing an easy-to-use OCR CLI
- 🌐 **[newbee-ocr-service](https://github.com/zibo-chen/newbee-ocr-service)** - HTTP service based on this library, providing RESTful API endpoints

## Model Download

All model files can be downloaded from:
- [PaddleOCR Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)
- Conversion tools are needed to convert PaddlePaddle models to MNN format

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - For providing the original OCR models and research
- [MNN](https://github.com/alibaba/MNN) - For providing the efficient neural network inference framework
