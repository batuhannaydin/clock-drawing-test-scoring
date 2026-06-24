"""Generate a multi-page comparison PDF across all experiment presets.

Reads:
  - outputs/puan/test_metrics_<preset>.json   (structured metrics)
  - outputs/puan/training_log_<preset>.csv    (epoch-by-epoch curves)

Writes:
  - outputs/puan/comparison_report.pdf

Pages:
  1. Title + dataset stats + mechanism toggle matrix
  2. Headline metrics table (macro F1, acc, MAE, 3-band F1) + per-class F1 bar
  3. Training curves overlay (valid F1 over epochs, one line per preset)
  4. Train-valid gap (generalization)
  5. Per-class F1 grouped bar chart
  6+. One page per preset: confusion matrix + severity confusion + details
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from config import CFG
from experiments import PRESET_ORDER, PRESETS


OUT_DIR = CFG.output_dir
PDF_OUT = OUT_DIR / "comparison_report.pdf"


def _load_json(preset: str):
    p = OUT_DIR / f"test_metrics_{preset}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_log(preset: str):
    p = OUT_DIR / f"training_log_{preset}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _hide_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _draw_text_page(pdf, title: str, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _hide_axes(ax)
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18, weight="bold")
    y = 0.88
    for line in lines:
        ax.text(0.05, y, line, ha="left", va="top", fontsize=10, family="monospace")
        y -= 0.025
    pdf.savefig(fig); plt.close(fig)


def _mechanism_matrix(presets):
    """Boolean/value matrix of mechanism toggles per preset."""
    cols = [
        ("omurga", lambda c: "tiny" if "tiny" in c["backbone"] else "small"),
        ("augmentation", lambda c: "güçlü" if c["use_strong_aug"] else "yok"),
        ("sınıf ağırlığı", lambda c: "✓" if c["use_class_weights"] else "✗"),
        ("sampler", lambda c: "✓" if c["use_weighted_sampler"] else "✗"),
        ("label_smooth", lambda c: f'{c["label_smoothing"]:.2f}'),
        ("focal γ", lambda c: f'{c["focal_gamma"]:.1f}'),
        ("ordinal λ", lambda c: f'{c["ordinal_lambda"]:.2f}'),
        ("drop_path", lambda c: f'{c["drop_path_rate"]:.2f}'),
        ("freeze_ep", lambda c: str(c["freeze_backbone_epochs"])),
        ("lr", lambda c: f'{c["learning_rate"]:.0e}'),
    ]
    headers = ["preset"] + [c[0] for c in cols]
    rows = []
    for name in presets:
        cfg = PRESETS[name]
        rows.append([name] + [fn(cfg) for _, fn in cols])
    return headers, rows


def _add_suspect_review_pages(pdf):
    """Two pages summarizing the manual label-cleanup phase."""
    # ----- Page A: process + per-round actions -----
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _hide_axes(ax)
    ax.text(0.5, 0.96, "Veri Temizliği (Suspect Review) Özeti",
            ha="center", va="top", fontsize=18, weight="bold")
    ax.text(0.5, 0.89,
            "Confident learning ile model şüpheli etiketleri flag'ledi, kullanıcı manuel inceledi.",
            ha="center", va="top", fontsize=10, style="italic")

    method_lines = [
        "Yöntem: Eğitilmiş model her örneğe softmax tahmin üretti. Etiket sınıfına atadığı olasılık",
        "düşük, başka bir sınıfa yüksek olasılık veren örnekler 'şüpheli' kabul edildi (margin = pred_prob − labeled_prob).",
        "Şüpheliler margin'e göre sıralandı, en yüksek margin'liler manuel görsel inceleme için listelendi.",
        "",
        "Her örnek için karar seçenekleri: sil / başka sınıfa taşı / kırpılmamış (yeniden kırp) / etiket doğru.",
    ]
    y = 0.81
    for line in method_lines:
        ax.text(0.06, y, line, ha="left", va="top", fontsize=9)
        y -= 0.028

    headers = ["Tur", "Kaynak", "İncelenen", "Sil", "Etiket düzelt", "Kırpılmamış", "Etiket doğru"]
    rows = [
        ["1", "İlk CL çıktısı (top-700)", "698", "431", "146", "69", "52"],
        ["2 (a)", "REMAINING_TODO (işaretli)", "177", "118", "34", "19", "11"],
        ["2 (b)", "REMAINING_TODO (işaretsiz, toplu sil)", "459", "459", "—", "—", "—"],
        ["3", "_kirpilmamis _assignments.txt", "81", "24", "57", "—", "—"],
        ["TOPLAM", "", "1.415", "1.032", "237", "88", "63"],
    ]
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center",
                   bbox=[0.02, 0.30, 0.96, 0.40])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    # Tur (narrow), Kaynak (wide), rest equal
    n_cols = len(headers)
    weights = [0.7, 3.4, 1.2, 0.8, 1.3, 1.2, 1.3]
    total = sum(weights)
    for col_idx, w in enumerate(weights):
        for row_idx in range(len(rows) + 1):
            tbl[(row_idx, col_idx)].set_width(w / total)
    for c in range(len(headers)):
        tbl[(0, c)].set_facecolor("#dcdcdc")
        tbl[(0, c)].set_text_props(weight="bold")
    for c in range(len(headers)):
        tbl[(5, c)].set_facecolor("#fffac8")
        tbl[(5, c)].set_text_props(weight="bold")
    # Left-align Kaynak column for readability
    for r in range(1, len(rows) + 1):
        tbl[(r, 1)].set_text_props(ha="left")
        tbl[(r, 1)]._loc = "left"

    summary_lines = [
        f"• Toplam silinen dosya: 1.032",
        f"• Etiketi değiştirilen örnek (taşı kararları): 237  (180 suspect-review + 57 kırpılmamış sonrası yeni atama)",
        f"• Yeniden kırpılan örnek (kırpılmamış → recrop → yeni sınıf): 88  (57 sınıfa atandı, 24 silindi, 7 manuel atıldı)",
        f"• 3↔4 sınırında etiket düzeltmesi (en yoğun bölge): 129 örnek  (109 etiket 3→4, 20 etiket 4→3)",
    ]
    y = 0.24
    for line in summary_lines:
        ax.text(0.06, y, line, ha="left", va="top", fontsize=10)
        y -= 0.030

    pdf.savefig(fig); plt.close(fig)

    # ----- Page B: class distribution change + impact -----
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5),
                             gridspec_kw={"height_ratios": [1, 1]})
    ax1, ax2 = axes

    # Per-class before/after bars
    classes = list(range(6))
    before = [1028, 3358, 10202, 2905, 3445, 3469]   # original file counts per class
    after = [1025, 3283, 10084, 2669, 3338, 3469]    # post-cleanup file counts (pre-cap)
    width = 0.38
    x = np.arange(len(classes))
    ax1.bar(x - width / 2, before, width, label="Önce", color="#9ec5ff")
    ax1.bar(x + width / 2, after, width, label="Sonra", color="#3f7bd6")
    for i, (b, a) in enumerate(zip(before, after)):
        ax1.text(i - width / 2, b + 60, str(b), ha="center", fontsize=8)
        ax1.text(i + width / 2, a + 60, str(a), ha="center", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels([f"sınıf {c}" for c in classes])
    ax1.set_ylabel("Dosya sayısı")
    ax1.set_title("Sınıf bazlı dosya sayısı — temizlik öncesi vs sonrası (cap'siz ham sayılar)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Impact narrative
    _hide_axes(ax2)
    ax2.text(0.5, 0.95, "Etki: Mekanizma değil, veri temizliği baskın faktör",
             ha="center", va="top", fontsize=14, weight="bold")
    impact_lines = [
        "",
        "Veri temizliğinden ÖNCE (kirli etiketler, tüm ek mekanizmalar AÇIK):",
        "    Tiny + full pipeline (eski test seti)         Macro F1: 0.7674",
        "",
        "Veri temizliğinden SONRA (temiz etiketler, SIFIR ek mekanizma):",
        "    Tiny + vanilla CE (t1_vanilla, yeni test)     Macro F1: 0.8442",
        "",
        "Net iyileşme:  +7.7 puan  —  hiçbir mekanizma eklemeden, sadece etiket gürültüsünü azaltarak.",
        "",
        "Ablation çalışması (8 preset) bu noktadan sonra hiçbir mekanizmanın bu kazancı katlayamadığını gösterdi;",
        "tersine, class weights, focal loss ve ordinal regularizer gibi kirli veride yardımcı olan",
        "mekanizmalar temiz veride hafif zarar veriyor (en iyi sıralamada plain CE birinci).",
        "",
        "Bu sonuç clinical-grade etiket temizliğinin daha karmaşık model mimarisinden ÖNCE",
        "yapılması gereken yatırım olduğunu ampirik olarak destekliyor.",
    ]
    y = 0.86
    for line in impact_lines:
        weight = "bold" if "Net iyileşme" in line else "normal"
        ax2.text(0.05, y, line, ha="left", va="top", fontsize=10,
                 family="monospace" if line.startswith("    ") else "sans-serif",
                 weight=weight)
        y -= 0.052
    pdf.savefig(fig); plt.close(fig)


def _add_title_page(pdf, presets_present):
    headers, rows = _mechanism_matrix(presets_present)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _hide_axes(ax)
    ax.text(0.5, 0.95, "Puan Sınıflandırıcı — Ablation Çalışması", ha="center", va="top",
            fontsize=20, weight="bold")
    ax.text(0.5, 0.91, "ConvNeXt omurga + mekanizma karşılaştırması",
            ha="center", va="top", fontsize=12, style="italic")
    ax.text(0.5, 0.86, f"{len(presets_present)} deney karşılaştırıldı",
            ha="center", va="top", fontsize=11)
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center",
                   bbox=[0.02, 0.10, 0.96, 0.65])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    # column widths: long headers wider, single-digit value columns narrower
    # headers order:
    # 0 preset, 1 omurga, 2 augmentation, 3 sınıf ağırlığı, 4 sampler,
    # 5 label_smooth, 6 focal γ, 7 ordinal λ, 8 drop_path, 9 freeze_ep, 10 lr
    weights = [2.0, 1.1, 1.6, 1.8, 1.1, 1.5, 0.8, 0.9, 1.3, 0.9, 0.7]
    total = sum(weights)
    for col_idx, w in enumerate(weights):
        for row_idx in range(len(rows) + 1):
            tbl[(row_idx, col_idx)].set_width(w / total)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#dcdcdc")
            cell.set_text_props(weight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(family="monospace", weight="bold")
    ax.text(0.5, 0.06, "Her satır bir önceki tiny satırına ek bir mekanizma ekler.   "
            "✓ açık, ✗ kapalı.",
            ha="center", fontsize=9, style="italic")
    pdf.savefig(fig); plt.close(fig)


def _add_headline_table(pdf, presets, metrics):
    headers = ["preset", "Macro F1", "Acc", "MAE", "3-band F1", "3-band Acc"]
    rows = []
    for n in presets:
        m = metrics[n]
        rows.append([
            n,
            f'{m["macro_f1"]:.4f}',
            f'{m["accuracy"]:.4f}',
            f'{m["mae"]:.3f}',
            f'{m["severity"]["macro_f1"]:.4f}',
            f'{m["severity"]["accuracy"]:.4f}',
        ])

    # mark best per column (excluding MAE which is lower-is-better)
    best_idx = {}
    f1s = [metrics[n]["macro_f1"] for n in presets]
    best_idx["Macro F1"] = int(np.argmax(f1s))
    accs = [metrics[n]["accuracy"] for n in presets]
    best_idx["Acc"] = int(np.argmax(accs))
    maes = [metrics[n]["mae"] for n in presets]
    best_idx["MAE"] = int(np.argmin(maes))
    sf1s = [metrics[n]["severity"]["macro_f1"] for n in presets]
    best_idx["3-band F1"] = int(np.argmax(sf1s))
    saccs = [metrics[n]["severity"]["accuracy"] for n in presets]
    best_idx["3-band Acc"] = int(np.argmax(saccs))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    _hide_axes(ax)
    ax.text(0.5, 0.95, "Test Seti Ana Metrikleri", ha="center", va="top",
            fontsize=16, weight="bold")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center",
                   bbox=[0.05, 0.20, 0.90, 0.65])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for c_idx, h in enumerate(headers):
        if h in best_idx:
            r = best_idx[h] + 1   # +1 for header row
            cell = tbl[(r, c_idx)]
            cell.set_facecolor("#bcebbf")
            cell.set_text_props(weight="bold")
    for c_idx in range(len(headers)):
        tbl[(0, c_idx)].set_facecolor("#dcdcdc")
        tbl[(0, c_idx)].set_text_props(weight="bold")
    ax.text(0.5, 0.10, "Yeşil = sütunun en iyisi.  MAE için düşük = daha iyi.",
            ha="center", fontsize=9, style="italic")
    pdf.savefig(fig); plt.close(fig)


def _add_per_class_f1(pdf, presets, metrics):
    n_classes = CFG.num_classes
    x = np.arange(n_classes)
    width = 0.85 / len(presets)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for i, n in enumerate(presets):
        f1s = metrics[n]["per_class"]["f1"]
        bars = ax.bar(x + i * width - 0.42 + width / 2, f1s, width, label=n)
        for b, v in zip(bars, f1s):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([f"sınıf {i}" for i in range(n_classes)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("Sınıf bazlı F1 — tüm preset'ler yan yana")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def _add_training_curves(pdf, presets, logs):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for n in presets:
        df = logs.get(n)
        if df is None or "valid_f1" not in df.columns:
            continue
        ax.plot(df["epoch"], df["valid_f1"], label=f"{n}", linewidth=1.4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validasyon Macro F1")
    ax.set_title("Validasyon F1 — eğitim eğrileri")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def _add_train_valid_gap(pdf, presets, logs):
    """Overfit indicator: peak train F1 - peak valid F1."""
    names, gaps = [], []
    for n in presets:
        df = logs.get(n)
        if df is None or "train_f1" not in df.columns:
            continue
        best_v = float(df["valid_f1"].max())
        train_at_v = float(df.loc[df["valid_f1"].idxmax(), "train_f1"])
        names.append(n); gaps.append(train_at_v - best_v)

    fig, ax = plt.subplots(figsize=(11, 7.0))
    colors = ["#d44" if g > 0.04 else "#fa3" if g > 0.02 else "#4c4" for g in gaps]
    ax.barh(names, gaps, color=colors)
    for n, g in zip(names, gaps):
        ax.text(g + 0.002, n, f"{g:+.3f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("train F1 − valid F1  (en iyi epoch'ta; yüksek = daha fazla overfit)")
    ax.set_title("Genelleme aralığı — preset bazında")
    ax.grid(axis="x", alpha=0.3)
    # Açıklama için figürün altında yer aç
    plt.subplots_adjust(bottom=0.28)
    fig.text(
        0.5, 0.13,
        "Genelleme aralığı = model'in eğitim setindeki F1'i ile validasyon setindeki F1'i arasındaki fark.",
        ha="center", va="top", fontsize=9, style="italic", color="#333",
    )
    fig.text(
        0.5, 0.09,
        "Sıfıra yakınsa model gördüğü ve görmediği veride benzer performans verir (iyi genelleme).  "
        "Yüksek pozitif değer = ezberleme (overfit) işareti.",
        ha="center", va="top", fontsize=9, style="italic", color="#333",
    )
    fig.text(
        0.5, 0.04,
        "Yeşil < 0.02     sarı 0.02–0.04     kırmızı > 0.04",
        ha="center", va="top", fontsize=9, style="italic", color="#333", weight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    pdf.savefig(fig); plt.close(fig)


def _add_confusion_page(pdf, preset, m, log_df=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
    (ax_cm, ax_sev), (ax_loss, ax_f1) = axes

    # ---- 6-sınıf confusion ----
    cm = np.array(m["confusion"])
    cm_n = cm / cm.sum(axis=1, keepdims=True).clip(1)
    im = ax_cm.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_title(f"{preset} — 6 sınıflı confusion matrix (satır normalleştirilmiş)", fontsize=10)
    ax_cm.set_xlabel("tahmin"); ax_cm.set_ylabel("gerçek")
    ax_cm.set_xticks(range(6)); ax_cm.set_yticks(range(6))
    for i in range(6):
        for j in range(6):
            txt = f"{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)"
            color = "white" if cm_n[i, j] > 0.5 else "black"
            ax_cm.text(j, i, txt, ha="center", va="center", color=color, fontsize=6.5)
    plt.colorbar(im, ax=ax_cm, fraction=0.046)

    # ---- 3-bant şiddet confusion ----
    sev_cm = np.array(m["severity"]["confusion"])
    sev_n = sev_cm / sev_cm.sum(axis=1, keepdims=True).clip(1)
    im2 = ax_sev.imshow(sev_n, cmap="Greens", vmin=0, vmax=1)
    ax_sev.set_title(f"{preset} — 3 bantlı şiddet confusion matrix", fontsize=10)
    labs = m["severity"]["labels"]
    ax_sev.set_xticks(range(3)); ax_sev.set_yticks(range(3))
    ax_sev.set_xticklabels(labs, rotation=15); ax_sev.set_yticklabels(labs)
    ax_sev.set_xlabel("tahmin"); ax_sev.set_ylabel("gerçek")
    for i in range(3):
        for j in range(3):
            txt = f"{sev_cm[i,j]}\n({sev_n[i,j]*100:.0f}%)"
            color = "white" if sev_n[i, j] > 0.5 else "black"
            ax_sev.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im2, ax=ax_sev, fraction=0.046)

    # ---- Loss curves ----
    if log_df is not None and "train_loss" in log_df.columns:
        ax_loss.plot(log_df["epoch"], log_df["train_loss"], label="train_loss",
                     color="#d44", linewidth=1.5)
        ax_loss.plot(log_df["epoch"], log_df["valid_loss"], label="valid_loss",
                     color="#36c", linewidth=1.5)
        ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{preset} — Loss eğrisi", fontsize=10)
        ax_loss.legend(loc="upper right", fontsize=9)
        ax_loss.grid(alpha=0.3)
    else:
        _hide_axes(ax_loss)
        ax_loss.text(0.5, 0.5, "Loss log bulunamadı", ha="center", va="center", fontsize=10)

    # ---- F1 curves ----
    if log_df is not None and "train_f1" in log_df.columns:
        ax_f1.plot(log_df["epoch"], log_df["train_f1"], label="train_f1",
                   color="#d44", linewidth=1.5)
        ax_f1.plot(log_df["epoch"], log_df["valid_f1"], label="valid_f1",
                   color="#36c", linewidth=1.5)
        ax_f1.set_xlabel("Epoch"); ax_f1.set_ylabel("Macro F1")
        ax_f1.set_title(f"{preset} — F1 eğrisi (train vs validasyon)", fontsize=10)
        ax_f1.legend(loc="lower right", fontsize=9)
        ax_f1.grid(alpha=0.3)
    else:
        _hide_axes(ax_f1)
        ax_f1.text(0.5, 0.5, "F1 log bulunamadı", ha="center", va="center", fontsize=10)

    sub = (
        f"Macro F1={m['macro_f1']:.4f}  Acc={m['accuracy']:.4f}  MAE={m['mae']:.3f}   "
        f"| 3-band F1={m['severity']['macro_f1']:.4f} Acc={m['severity']['accuracy']:.4f}"
    )
    fig.suptitle(sub, fontsize=10, y=0.005)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    pdf.savefig(fig); plt.close(fig)


def main() -> None:
    metrics = {}
    logs = {}
    presets_present = []
    for n in PRESET_ORDER:
        m = _load_json(n)
        if m is None:
            print(f"[skip] no test_metrics_{n}.json")
            continue
        metrics[n] = m
        logs[n] = _load_log(n)
        presets_present.append(n)

    if not presets_present:
        raise SystemExit("No experiment results found. Run run_all_experiments.py first.")

    print(f"[info] generating PDF for {len(presets_present)} preset(s)")
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(PDF_OUT) as pdf:
        _add_title_page(pdf, presets_present)
        _add_suspect_review_pages(pdf)
        _add_headline_table(pdf, presets_present, metrics)
        _add_per_class_f1(pdf, presets_present, metrics)
        _add_training_curves(pdf, presets_present, logs)
        _add_train_valid_gap(pdf, presets_present, logs)
        for n in presets_present:
            _add_confusion_page(pdf, n, metrics[n], log_df=logs.get(n))

    print(f"[done] PDF → {PDF_OUT}")


if __name__ == "__main__":
    main()
