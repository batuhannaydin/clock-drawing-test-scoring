"""Generate a separate PDF with dataset details + t2_aug model architecture.

Output: outputs/puan/dataset_and_model_report.pdf

Pages:
  1. Başlık + görev tanımı (CDT)
  2. Ham arşiv: kaynak, dosya formatı, sınıf dağılımı
  3. Aktif veri seti: kırpma pipeline + temizlik sonrası dağılım
  4. Train/valid/test split + 3-bantlı şiddet etiketi
  5. Seçilen model: t2_aug — neden bu?
  6. ConvNeXt-Tiny mimari özeti (stages, depths, dims)
  7. t2_aug eğitim konfigürasyonu (hiperparametreler, augmentation)
  8. t2_aug eğitim dinamikleri (loss + F1 curves)
  9. t2_aug test sonuçları (confusion matrices + metrikler)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from config import CFG
from experiments import PRESETS


OUT_DIR = CFG.output_dir
PDF_OUT = OUT_DIR / "dataset_and_model_report.pdf"
PRESET = "t2_aug"


# ============================================================
# Helpers
# ============================================================
def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _add_text_block(ax, x, y, lines, fontsize=9, family=None, lh=0.028):
    """Write a list of lines starting at (x, y), each below the previous."""
    cur = y
    for line in lines:
        kw = {"ha": "left", "va": "top", "fontsize": fontsize}
        if family:
            kw["family"] = family
        ax.text(x, cur, line, **kw)
        cur -= lh
    return cur


# ============================================================
# Page 1: title + task
# ============================================================
def _page_title(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _hide(ax)
    ax.text(0.5, 0.92, "Veri Seti ve Model Mimarisi",
            ha="center", va="top", fontsize=22, weight="bold")
    ax.text(0.5, 0.86,
            "Saat Çizim Testi (CDT) 6-Sınıflı Puan Sınıflandırması",
            ha="center", va="top", fontsize=14, style="italic")
    ax.text(0.5, 0.81,
            "Ek belge — ana karşılaştırma raporunun yanına eklenir",
            ha="center", va="top", fontsize=10, color="#666")

    intro = [
        "Görev nedir?",
        "    Saat Çizim Testi (Clock Drawing Test, CDT), yaşlı bireylerde bilişsel durumu değerlendirmek",
        "    için kullanılan basit ama klinik olarak güçlü bir tarama aracıdır. Hastadan belirli bir saati",
        "    (örn. 11:10) bir kağıda çizmesi istenir; doktor sonucu 0-5 ölçeğinde puanlar.",
        "",
        "Bu projenin amacı:",
        "    El ile çizilmiş saat görüntülerinden, klinik puanı (0-5) otomatik tahmin eden derin öğrenme",
        "    tabanlı bir sınıflandırıcı geliştirmek. Hedef: insan değerlendirmesinin hızlı dijital eşdeğerini",
        "    sunarak büyük ölçekli tarama çalışmalarını kolaylaştırmak.",
        "",
        "Puan ölçeği (klinik anlam):",
        "    Puan 0  →  Şiddetli bilişsel kayıp şüphesi (saat hiç tanınmıyor)",
        "    Puan 1  →  Ciddi bozulma",
        "    Puan 2  →  Belirgin bozulma",
        "    Puan 3  →  Orta bozulma",
        "    Puan 4  →  Hafif bozulma / sınırda",
        "    Puan 5  →  Sağlıklı çizim",
        "",
        "Klinik 3-bant ayrımı (asıl karar sınırı):",
        "    Şiddetli (0-1)   ·   Orta (2-3)   ·   Sağlıklı (4-5)",
    ]
    _add_text_block(ax, 0.06, 0.72, intro, fontsize=10, lh=0.026)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 2: raw archive
# ============================================================
def _page_raw_archive(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Ham Arşiv (master_etiketler.csv)", fontsize=16, weight="bold", y=0.96)

    # Top-left: facts table
    ax_facts = fig.add_axes([0.05, 0.55, 0.42, 0.32])
    _hide(ax_facts)
    headers = ["Özellik", "Değer"]
    rows = [
        ["Toplam çizim", "59.417"],
        ["Tekil özne (spid)", "13.642"],
        ["Round (tur)", "1 — 12"],
        ["Kişi başına ort. çizim", "4.36 (medyan 3)"],
        ["Dosya formatı", "TIFF (ham)"],
        ["Dosya adı şablonu", "<spid>_R<round>.tif"],
        ["Örnek", "10000003_R1.tif"],
        ["Etiket kolonları", "dosya_adi, spid, round, puan"],
    ]
    tbl = ax_facts.table(cellText=rows, colLabels=headers, loc="center",
                         cellLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for c in range(2):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")
    ax_facts.set_title("Temel sayılar", fontsize=11, weight="bold", pad=8)

    # Top-right: class distribution bar
    ax_dist = fig.add_axes([0.55, 0.55, 0.42, 0.32])
    classes = list(range(6))
    counts = [540, 1957, 6597, 12529, 22080, 15714]
    pcts = [c / sum(counts) * 100 for c in counts]
    bars = ax_dist.bar(classes, counts, color="#3f7bd6")
    for b, n, p in zip(bars, counts, pcts):
        ax_dist.text(b.get_x() + b.get_width() / 2,
                     b.get_height() + 300,
                     f"{n:,}\n(%{p:.1f})", ha="center", fontsize=8)
    ax_dist.set_xlabel("Puan")
    ax_dist.set_ylabel("Çizim sayısı")
    ax_dist.set_title("Ham arşivde sınıf dağılımı", fontsize=11, weight="bold")
    ax_dist.set_ylim(0, max(counts) * 1.22)
    ax_dist.grid(axis="y", alpha=0.3)

    # Bottom: narrative
    ax_note = fig.add_axes([0.05, 0.05, 0.92, 0.43])
    _hide(ax_note)
    narrative = [
        "Veri kaynağı:",
        "    Çalışmada kullanılan saat çizimleri longitüdinal bir bilişsel sağlık takip projesinden geldi.",
        "    Her özne (subject) bir veya birkaç farklı 'round' (zaman içinde tekrarlanan ölçüm) ile",
        "    saat çizdi; bu yüzden tek bir spid'ye birden çok çizim düşebilir (ortalama 4.36).",
        "",
        "Sınıf dengesizliği (kritik nokta):",
        "    Ham arşivde puan 4 ve 5 (sağlıklı/yarı-sağlıklı) toplamın %63.6'sı; puan 0 sadece %0.9.",
        "    En sık karşılaşılan puan 4 (22.080 çizim), en nadir puan 0 (540 çizim). Bu 40:1'i geçen",
        "    dengesizlik, eğitim sırasında özel önlem gerektirir (per-class cap, ağırlıklı kayıp gibi).",
        "",
        "Etiketleme:",
        "    Tüm puanlar arşivde önceden hazır (manuel klinik puanlama). Biz bu etiketleri yeniden",
        "    üretmedik — sadece düzelttik (suspect review süreciyle 237 etiket).",
    ]
    _add_text_block(ax_note, 0.02, 0.95, narrative, fontsize=10, lh=0.060)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 3: active subset + preprocessing
# ============================================================
def _page_active(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Aktif Veri Seti (Ön İşlenmiş + Temizlenmiş)",
                 fontsize=16, weight="bold", y=0.96)

    # Top-left: preprocessing pipeline (text)
    ax_pipe = fig.add_axes([0.05, 0.55, 0.42, 0.35])
    _hide(ax_pipe)
    ax_pipe.set_title("Ön işleme adımları", fontsize=11, weight="bold", pad=8)
    steps = [
        "1. TIFF dosyası okunur (resimler/<stem>.tif)",
        "",
        "2. Otomatik clock-bounding:",
        "    crop_clock() saat bölgesini bulur,",
        "    form metni ve kenar gürültüsü temizlenir",
        "",
        "3. 224 × 224 piksele yeniden boyutlandırılır",
        "",
        "4. Gri seviye (L modu) PNG olarak kaydedilir",
        "    processed/<puan>/<stem>.png",
        "",
        "5. Eğitim sırasında 3 kanallı RGB'ye dönüştürülür",
        "    (ImageNet pretrain'i için)",
        "",
        "6. ImageNet ortalaması ve std ile normalize edilir",
    ]
    _add_text_block(ax_pipe, 0.02, 0.95, steps, fontsize=9, family="monospace", lh=0.052)

    # Top-right: post-cleanup class distribution
    ax_dist = fig.add_axes([0.55, 0.55, 0.42, 0.35])
    classes = list(range(6))
    counts = [1025, 3283, 10084, 2669, 3338, 3469]
    bars = ax_dist.bar(classes, counts, color="#3f7bd6")
    for b, n in zip(bars, counts):
        ax_dist.text(b.get_x() + b.get_width() / 2, b.get_height() + 100,
                     f"{n:,}", ha="center", fontsize=9)
    ax_dist.set_xlabel("Puan")
    ax_dist.set_ylabel("Aktif örnek sayısı")
    ax_dist.set_title("Temizlik sonrası aktif örnekler", fontsize=11, weight="bold")
    ax_dist.set_ylim(0, max(counts) * 1.15)
    ax_dist.grid(axis="y", alpha=0.3)

    # Bottom: facts + note
    ax_note = fig.add_axes([0.05, 0.05, 0.92, 0.43])
    _hide(ax_note)
    narrative = [
        f"Ham arşivin (59.417) sadece 23.868'i (%40) başarıyla otomatik kırpıldı ve PNG olarak hazırlandı.",
        f"Geri kalan TIFF'ler ya başarısız kırpıldı, ya kırpma kalitesi düşüktü, ya da projeye dahil edilmedi.",
        "",
        f"Bu 23.868 örnek, üzerine 3 turlu manuel veri temizliği uygulandı:",
        f"    • 1.032 örnek silindi (kırpılma sorunu, yanlış etiket vb.)",
        f"    • 237 örneğin etiketi düzeltildi (en yoğun: 109 örnek '3 → 4', 20 örnek '4 → 3')",
        f"    • 88 örnek yeniden kırpıldı (kötü ilk kırpma; 57'si yeni sınıfa atandı, 24'ü silindi)",
        "",
        "Sonuç:",
        "    • Sınıf 0 (en nadir) ve 5 hemen hemen aynı kaldı — orada gürültü zaten azdı",
        "    • Sınıf 3 belirgin azaldı (2.905 → 2.669) — 3↔4 sınırındaki etiket düzeltmesi nedeniyle",
        "    • Toplam aktif örnek: 23.868 (sınıf 2 hâlâ dominant, %42)",
    ]
    _add_text_block(ax_note, 0.02, 0.95, narrative, fontsize=10, lh=0.060)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 4: split + 3-band
# ============================================================
def _page_split(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Eğitim Bölünmesi ve 3-Bantlı Şiddet Etiketi",
                 fontsize=16, weight="bold", y=0.96)

    # Top: split table
    ax_tbl = fig.add_axes([0.10, 0.62, 0.80, 0.28])
    _hide(ax_tbl)
    headers = ["Split", "Toplam", "0", "1", "2", "3", "4", "5"]
    rows = [
        ["Train", "10.440", "277", "828", "2.400", "2.135", "2.400", "2.400"],
        ["Valid", "1.306", "35", "104", "300", "267", "300", "300"],
        ["Test", "1.306", "35", "104", "300", "267", "300", "300"],
        ["TOPLAM", "13.052", "347", "1.036", "3.000", "2.669", "3.000", "3.000"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="center",
                       cellLoc="center", bbox=[0, 0, 1, 0.85])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(4, c)].set_facecolor("#fffac8")
        tbl[(4, c)].set_text_props(weight="bold")
    ax_tbl.text(0.5, 0.96, "Per-class cap = 3.000  (sınıf 2/4/5 tavanlandı)",
                ha="center", fontsize=9, style="italic")

    # Bottom: 3-band table + narrative
    ax_note = fig.add_axes([0.05, 0.05, 0.92, 0.50])
    _hide(ax_note)
    band_lines = [
        "3-Bantlı Şiddet Etiketi (ikincil ölçüm)",
        "",
        "    Şiddetli   (puan 0-1)   →   belirgin bilişsel kayıp şüphesi",
        "    Orta       (puan 2-3)   →   sınırda durum",
        "    Sağlıklı   (puan 4-5)   →   normal çizim",
        "",
        "Bu üç-bant ayrımı klinik karar verme açısından asıl önemli bölünmedir:",
        "    4 ↔ 5 karışıklığı klinik olarak zararsız (her ikisi de 'sağlıklı' bandında)",
        "    1 ↔ 2 karışıklığı ise ciddi yanlış (şiddetli → orta yanlış sınıflandırma)",
        "",
        "Modeller hem 6-sınıflı macro F1 hem de 3-bantlı F1 üzerinden raporlanır.",
        "",
        "Split mantığı:",
        "    • Stratified: her sınıf train/valid/test'te orantılı temsil edilir",
        "    • Per-class cap: 3.000 (sınıf 2 normalde 10.084 — 3.000'e indirildi)",
        "    • test_split = 0.10, valid_split = 0.10, train = kalan 0.80",
        "    • Seed = 42, kalıcı: split.csv'de saklanır, train/evaluate aynı bölünmeyi paylaşır",
    ]
    _add_text_block(ax_note, 0.02, 0.95, band_lines, fontsize=10, lh=0.055)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 5: Why t2_aug
# ============================================================
def _page_why_t2(pdf, t2_metrics):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _hide(ax)
    ax.text(0.5, 0.94, "Seçilen Model: t2_aug",
            ha="center", va="top", fontsize=20, weight="bold")
    ax.text(0.5, 0.89,
            "Tiny + güçlü augmentation, başka mekanizma yok",
            ha="center", va="top", fontsize=12, style="italic")

    m = t2_metrics
    headline = [
        f"Macro F1:           {m['macro_f1']:.4f}",
        f"Accuracy:           {m['accuracy']:.4f}      ← ablation içinde EN İYİ",
        f"MAE:                {m['mae']:.3f}        ← ablation içinde EN İYİ",
        f"3-bant F1:          {m['severity']['macro_f1']:.4f}",
        f"3-bant Accuracy:    {m['severity']['accuracy']:.4f}      ← ablation içinde EN İYİ",
    ]
    _add_text_block(ax, 0.20, 0.81, headline,
                    fontsize=11, family="monospace", lh=0.030)

    rationale = [
        "Neden bu model? — 4 ana gerekçe",
        "",
        "1. Çoklu metrik şampiyonu",
        "    Test setinde Acc, MAE ve 3-bant Acc kategorilerinde en yüksek skoru aldı.",
        "    Macro F1'de yalnız 0.001 farkla 2. — bu istatistiksel olarak anlamsız.",
        "",
        "2. MAE en düşük → ordinal görev için kritik",
        "    Puan ordinal (sıralı) bir skor. MAE 'yanlış olduğunda ne kadar yanlış?' sorusunun cevabı.",
        "    t2_aug = 0.142, en kötü = 0.166. %14 daha küçük 'yanılma mesafesi'.",
        "",
        "3. Klinik anlamlılık (3-bant)",
        "    Asıl klinik karar Şiddetli / Orta / Sağlıklı bandında verilir. t2_aug bu boyutta açık ara.",
        "",
        "4. Bilimsel savunulabilirlik",
        "    Augmentation standart bir tekniktir, savunulabilir. Daha karmaşık mekanizmalar (focal,",
        "    ordinal, class weights) ablation içinde t2_aug'u geçemedi → 'iyi veri > karmaşık model'",
        "    mesajını destekler. Bu seçim 'data-centric AI' literatürüne uyumludur.",
    ]
    _add_text_block(ax, 0.06, 0.60, rationale, fontsize=10, lh=0.030)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 6: ConvNeXt-Tiny architecture
# ============================================================
def _page_architecture(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Model Mimarisi: ConvNeXt-Tiny", fontsize=16, weight="bold", y=0.96)

    # Top: stages table
    ax_tbl = fig.add_axes([0.08, 0.60, 0.84, 0.30])
    _hide(ax_tbl)
    headers = ["Aşama", "Blok sayısı", "Kanal (dim)", "Çıkış çözünürlüğü", "Operasyon"]
    rows = [
        ["Stem",  "—", "96",  "56 × 56",  "Conv 4×4, stride 4  +  LayerNorm"],
        ["Stage 1", "3", "96",  "56 × 56",  "ConvNeXt blok (DW 7×7 + 2-katmanlı MLP)"],
        ["Stage 2", "3", "192", "28 × 28",  "Downsample + ConvNeXt blok"],
        ["Stage 3", "9", "384", "14 × 14",  "Downsample + ConvNeXt blok"],
        ["Stage 4", "3", "768", "7 × 7",    "Downsample + ConvNeXt blok"],
        ["Head", "—", "768 → 6", "—", "GlobalAvgPool + Dropout(0.4) + Linear(768→6)"],
    ]
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="center",
                       cellLoc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")
    # Left-align last column
    for r in range(1, len(rows) + 1):
        tbl[(r, len(headers) - 1)]._loc = "left"

    ax_note = fig.add_axes([0.05, 0.04, 0.92, 0.50])
    _hide(ax_note)
    notes = [
        "ConvNeXt-Tiny: ViT'in tasarım fikirlerini saf CNN üzerine taşıyan modern bir mimari (Liu et al., 2022).",
        "Bu projede timm kütüphanesinden 'convnext_tiny.fb_in22k_ft_in1k' varyantı kullanıldı — ImageNet-22k'da",
        "pretrain edilip ImageNet-1k'da finetune edilmiş ağırlıklar.",
        "",
        "Toplam parametre: 27.823.046 (~27.8M)",
        "    • Backbone: 27.818.592",
        "    • Head: 4.614 (Dropout + Linear(768, 6))",
        "",
        "ConvNeXt blok yapısı (her blok):",
        "    DW 7×7 conv     (depthwise convolution, geniş receptive field)",
        "    LayerNorm       (BatchNorm yerine — Transformer'lardaki gibi)",
        "    Pointwise 1×1   (4x kanal genişletme: 96 → 384)",
        "    GELU            (aktivasyon)",
        "    Pointwise 1×1   (4x kanal daraltma: 384 → 96)",
        "    Residual connection",
        "",
        "Stage geçişlerinde 2×2 stride-2 conv ile downsampling + LayerNorm.",
        "Çıkış 7×7×768 feature map → global average pooling → 768-d vektör → head → 6 logit.",
    ]
    _add_text_block(ax_note, 0.02, 0.95, notes, fontsize=9.5, lh=0.045)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 7: t2_aug training config
# ============================================================
def _page_train_config(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("t2_aug Eğitim Konfigürasyonu", fontsize=16, weight="bold", y=0.96)

    # Top-left: hyperparameters
    ax_hp = fig.add_axes([0.05, 0.50, 0.43, 0.40])
    _hide(ax_hp)
    ax_hp.set_title("Hiperparametreler", fontsize=11, weight="bold", pad=4)
    headers = ["Parametre", "Değer"]
    rows = [
        ["image_size", "224 × 224"],
        ["batch_size", "32"],
        ["num_workers", "4"],
        ["epochs (üst sınır)", "50"],
        ["optimizer", "AdamW"],
        ["learning_rate (backbone)", "1e-4"],
        ["learning_rate (head)", "5e-4"],
        ["weight_decay", "5e-4"],
        ["dropout (head)", "0.4"],
        ["drop_path_rate", "0.0 (kapalı)"],
        ["freeze_backbone_epochs", "0 (donma yok)"],
        ["LR scheduler", "ReduceLROnPlateau"],
        ["    factor", "0.3"],
        ["    patience", "2"],
        ["    min_lr", "1e-6"],
        ["early stop patience", "6"],
        ["best metric", "valid macro F1"],
        ["seed", "42"],
    ]
    tbl = ax_hp.table(cellText=rows, colLabels=headers, loc="center",
                      cellLoc="left", bbox=[0, 0, 1, 0.95])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for c in range(2):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")

    # Top-right: augmentation
    ax_aug = fig.add_axes([0.52, 0.50, 0.43, 0.40])
    _hide(ax_aug)
    ax_aug.set_title("Eğitim Augmentation pipeline", fontsize=11, weight="bold", pad=4)
    aug_lines = [
        "1.  Resize(256, 256)",
        "2.  RandomCrop(224, 224)",
        "3.  RandomHorizontalFlip(p=0.5)",
        "4.  RandomRotation(degrees=20)",
        "5.  ColorJitter(brightness=0.3, contrast=0.3)",
        "6.  RandomAffine(",
        "        translate=(0.08, 0.08),",
        "        scale=(0.9, 1.1),",
        "        shear=5",
        "    )",
        "7.  RandAugment(num_ops=2, magnitude=7)",
        "8.  ToTensor()",
        "9.  Normalize(",
        "        mean=(0.485, 0.456, 0.406),",
        "        std=(0.229, 0.224, 0.225)",
        "    )",
        "10. RandomErasing(p=0.5, scale=(0.02, 0.2))",
        "",
        "Test/valid: yalnız Resize(224) + ToTensor + Normalize",
    ]
    _add_text_block(ax_aug, 0.02, 0.92, aug_lines,
                    fontsize=8.5, family="monospace", lh=0.045)

    # Bottom: loss + sampling notes
    ax_note = fig.add_axes([0.05, 0.05, 0.92, 0.42])
    _hide(ax_note)
    notes = [
        "Kayıp fonksiyonu (loss):",
        "    Plain Cross-Entropy  (FocalCrossEntropy γ=0, ordinal_lambda=0)",
        "    Sınıf ağırlıkları KAPALI — modele tüm sınıflar eşit önemde sunulur.",
        "    Label smoothing KAPALI.",
        "",
        "Örnekleme:",
        "    Standart shuffle  (WeightedRandomSampler kullanılmadı)",
        "    drop_last=True — son eksik batch atılır",
        "",
        "Eğitim akışı:",
        "    1. Backbone donmuyor — her epoch'tan itibaren tüm parametreler güncellenir",
        "    2. ReduceLROnPlateau valid F1'i 2 epoch boyunca düşmezse LR'yi 0.3× azaltır",
        "    3. Early stop: valid F1 6 epoch boyunca iyileşmezse eğitim biter",
        "    4. En iyi valid F1 noktasındaki ağırlıklar best_model_t2_aug.pth olarak kaydedilir",
    ]
    _add_text_block(ax_note, 0.02, 0.95, notes, fontsize=10, lh=0.060)
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 8: training dynamics
# ============================================================
def _page_training_curves(pdf, log_df):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    ax1, ax2 = axes
    fig.suptitle("t2_aug Eğitim Dinamikleri", fontsize=16, weight="bold", y=0.97)

    ax1.plot(log_df["epoch"], log_df["train_loss"], label="train_loss",
             color="#d44", linewidth=1.6)
    ax1.plot(log_df["epoch"], log_df["valid_loss"], label="valid_loss",
             color="#36c", linewidth=1.6)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss eğrisi (train vs validasyon)", fontsize=11)
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.3)

    ax2.plot(log_df["epoch"], log_df["train_f1"], label="train_f1",
             color="#d44", linewidth=1.6)
    ax2.plot(log_df["epoch"], log_df["valid_f1"], label="valid_f1",
             color="#36c", linewidth=1.6)
    best_ep = int(log_df.loc[log_df["valid_f1"].idxmax(), "epoch"])
    best_v = float(log_df["valid_f1"].max())
    ax2.axvline(best_ep, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(best_ep, best_v, f"  best epoch {best_ep}\n  valid_f1 = {best_v:.4f}",
             fontsize=9, va="bottom", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 eğrisi (train vs validasyon)", fontsize=11)
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)

    total_epochs = int(log_df["epoch"].max())
    train_at_best = float(log_df.loc[log_df["valid_f1"].idxmax(), "train_f1"])
    gap = train_at_best - best_v
    summary = (
        f"Toplam {total_epochs} epoch koşuldu  ·  En iyi valid F1 epoch {best_ep}'da yakalandı.   "
        f"Genelleme aralığı: train F1 {train_at_best:.3f} − valid F1 {best_v:.3f} = {gap:+.3f}  "
        f"({'iyi genelleme' if abs(gap) < 0.02 else 'hafif overfit' if gap < 0.04 else 'belirgin overfit'})."
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Page 9: test results
# ============================================================
def _page_test_results(pdf, m):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("t2_aug Test Seti Sonuçları", fontsize=16, weight="bold", y=0.97)

    # Top-left: 6-class confusion
    ax_cm = fig.add_axes([0.05, 0.42, 0.42, 0.45])
    cm = np.array(m["confusion"])
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(1)
    im = ax_cm.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title("6 sınıflı confusion (satır normalleştirilmiş)", fontsize=10)
    ax_cm.set_xlabel("tahmin"); ax_cm.set_ylabel("gerçek")
    ax_cm.set_xticks(range(6)); ax_cm.set_yticks(range(6))
    for i in range(6):
        for j in range(6):
            txt = f"{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)"
            color = "white" if cm_n[i, j] > 0.5 else "black"
            ax_cm.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im, ax=ax_cm, fraction=0.046)

    # Top-right: 3-band confusion
    ax_sev = fig.add_axes([0.55, 0.42, 0.42, 0.45])
    sev_cm = np.array(m["severity"]["confusion"])
    sev_n = sev_cm / sev_cm.sum(axis=1, keepdims=True).clip(1)
    im2 = ax_sev.imshow(sev_n, cmap="Greens", vmin=0, vmax=1)
    ax_sev.set_title("3 bantlı şiddet confusion", fontsize=10)
    labs = m["severity"]["labels"]
    ax_sev.set_xticks(range(3)); ax_sev.set_yticks(range(3))
    ax_sev.set_xticklabels(labs, rotation=15); ax_sev.set_yticklabels(labs)
    ax_sev.set_xlabel("tahmin"); ax_sev.set_ylabel("gerçek")
    for i in range(3):
        for j in range(3):
            txt = f"{sev_cm[i,j]}\n({sev_n[i,j]*100:.0f}%)"
            color = "white" if sev_n[i, j] > 0.5 else "black"
            ax_sev.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im2, ax=ax_sev, fraction=0.046)

    # Bottom: per-class table
    ax_tbl = fig.add_axes([0.10, 0.05, 0.80, 0.30])
    _hide(ax_tbl)
    pc = m["per_class"]
    headers = ["Sınıf", "Precision", "Recall", "F1", "Support"]
    rows = []
    for i in range(6):
        rows.append([str(i),
                     f'{pc["precision"][i]:.4f}',
                     f'{pc["recall"][i]:.4f}',
                     f'{pc["f1"][i]:.4f}',
                     str(pc["support"][i])])
    rows.append([
        "MACRO",
        "—",
        "—",
        f'{m["macro_f1"]:.4f}',
        f'{m["n_test"]} (toplam)'
    ])
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="center",
                       cellLoc="center", bbox=[0, 0, 1, 0.95])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(7, c)].set_facecolor("#fffac8")
        tbl[(7, c)].set_text_props(weight="bold")
    ax_tbl.set_title("Sınıf bazlı metrikler (test setinde)", fontsize=11, weight="bold", pad=4)

    pdf.savefig(fig); plt.close(fig)


# ============================================================
# Main
# ============================================================
def main() -> None:
    t2_json = OUT_DIR / f"test_metrics_{PRESET}.json"
    t2_log = OUT_DIR / f"training_log_{PRESET}.csv"
    if not t2_json.exists():
        raise SystemExit(f"Missing t2_aug results: {t2_json}")
    metrics = json.loads(t2_json.read_text(encoding="utf-8"))
    log_df = pd.read_csv(t2_log) if t2_log.exists() else None

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_OUT) as pdf:
        _page_title(pdf)
        _page_raw_archive(pdf)
        _page_active(pdf)
        _page_split(pdf)
        _page_why_t2(pdf, metrics)
        _page_architecture(pdf)
        _page_train_config(pdf)
        if log_df is not None:
            _page_training_curves(pdf, log_df)
        _page_test_results(pdf, metrics)

    print(f"[done] PDF → {PDF_OUT}")


if __name__ == "__main__":
    main()
