from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
NON_COMPLIANCE_KEYWORDS = [
    "sem",
    "no_",
    "no-",
    "without",
    "nao",
    "not_",
    "missing",
    "inadequado",
    "improper",
]

PPE_KEYWORDS = {
    "helmet": ["helmet", "capacete", "hardhat"],
    "vest": ["vest", "colete", "reflective"],
    "gloves": ["glove", "luva"],
    "goggles": ["goggle", "oculos", "glasses"],
    "mask": ["mask", "mascara", "respirator"],
    "boots": ["boot", "bota"],
}


@dataclass
class DetectionItem:
    class_name: str
    confidence: float


@dataclass
class ComplianceReport:
    timestamp: str
    source: str
    status: str
    total_detections: int
    non_compliant_count: int
    compliant_score: float
    summary: str


def runtime_base_dir() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return BASE_DIR


def resolve_model_path() -> Path:
    env_path = os.getenv("YOLO_MODEL_PATH")
    if env_path:
        model_path = Path(env_path).expanduser().resolve()
        if model_path.exists():
            return model_path

    candidates = [
        runtime_base_dir() / "yolo8n_v8" / "weights" / "best.pt",
        BASE_DIR / "yolo8n_v8" / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Nao foi possivel localizar o modelo 'best.pt'.\n"
        "Verifique se o arquivo existe em um destes caminhos:\n"
        f"{searched}\n"
        "Ou defina a variavel de ambiente YOLO_MODEL_PATH."
    )


MODEL_PATH = resolve_model_path()
_model = None
_model_lock = threading.Lock()


def load_model() -> YOLO:
    global _model
    with _model_lock:
        if _model is None:
            _model = YOLO(str(MODEL_PATH))
    return _model


def _contains_keyword(value: str, keywords: list[str]) -> bool:
    low = value.lower()
    return any(k in low for k in keywords)


def classify_compliance(detections: list[DetectionItem]) -> tuple[str, int, float, str]:
    total = len(detections)
    non_compliant = [
        det for det in detections if _contains_keyword(det.class_name, NON_COMPLIANCE_KEYWORDS)
    ]
    non_compliant_count = len(non_compliant)

    if total == 0:
        return "SEM DETECCOES", 0, 0.0, "Nenhum objeto detectado no frame analisado."

    if non_compliant_count > 0:
        status = "NAO CONFORME"
    else:
        status = "CONFORME"

    score = max(0.0, 100.0 * (1.0 - (non_compliant_count / total)))

    detected_names = [d.class_name.lower() for d in detections]
    present_ppe = []
    for ppe_name, keywords in PPE_KEYWORDS.items():
        if any(_contains_keyword(name, keywords) for name in detected_names):
            present_ppe.append(ppe_name)

    non_labels = ", ".join(sorted({d.class_name for d in non_compliant})) or "nenhuma"
    ppe_str = ", ".join(present_ppe) if present_ppe else "nenhum EPI identificado por palavra-chave"

    summary = (
        f"Status {status}. Total de deteccoes: {total}. "
        f"Classes de nao conformidade: {non_labels}. "
        f"EPIs aparentes no frame: {ppe_str}."
    )
    return status, non_compliant_count, score, summary


def run_detection(img_rgb: np.ndarray, conf: float) -> tuple[Image.Image, list[DetectionItem], ComplianceReport]:
    mdl = load_model()
    result = mdl(img_rgb, conf=float(conf))[0]
    annotated = Image.fromarray(np.ascontiguousarray(result.plot()))

    detections: list[DetectionItem] = []
    names = result.names
    for box in result.boxes:
        class_idx = int(box.cls.item())
        class_name = names[class_idx] if isinstance(names, list) else names.get(class_idx, str(class_idx))
        detections.append(DetectionItem(class_name=class_name, confidence=float(box.conf.item())))

    status, non_count, score, summary = classify_compliance(detections)
    report = ComplianceReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source="desconhecido",
        status=status,
        total_detections=len(detections),
        non_compliant_count=non_count,
        compliant_score=score,
        summary=summary,
    )
    return annotated, detections, report


class YOLOTkApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("EPI Compliance Detector")
        self.root.geometry("1280x780")
        self.root.minsize(1150, 720)
        self.root.configure(bg="#07131F")

        self.conf = tk.DoubleVar(value=0.30)
        self.current_image: Image.Image | None = None
        self.current_report: ComplianceReport | None = None
        self.reports_history: list[ComplianceReport] = []
        self.processing_lock = threading.Lock()
        self.countdown_running = False
        self.last_source = "arquivo"

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#07131F")
        container.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        left = tk.Frame(container, bg="#0B1E30", width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = tk.Frame(container, bg="#07131F")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 0))

        self._build_sidebar(left)
        self._build_main(right)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="EPI Vision",
            bg="#0B1E30",
            fg="#F3F6FA",
            font=("Segoe UI Semibold", 20),
        ).pack(anchor="w", padx=16, pady=(18, 2))

        tk.Label(
            parent,
            text="Deteccao e relatorio de conformidade",
            bg="#0B1E30",
            fg="#96A9BD",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=16, pady=(0, 20))

        self._styled_button(parent, "Abrir imagem", self.open_image, "#007A7A").pack(fill=tk.X, padx=16, pady=6)
        self._styled_button(parent, "Capturar webcam (3s)", self.capture_webcam, "#005D9A").pack(fill=tk.X, padx=16, pady=6)
        self._styled_button(parent, "Salvar imagem anotada", self.save_image, "#0B7D4B").pack(fill=tk.X, padx=16, pady=6)
        self._styled_button(parent, "Exportar relatorio CSV", self.export_report_csv, "#5B3C99").pack(fill=tk.X, padx=16, pady=6)

        tk.Label(
            parent,
            text="Confianca minima",
            bg="#0B1E30",
            fg="#DDE8F5",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=16, pady=(20, 4))

        tk.Scale(
            parent,
            variable=self.conf,
            from_=0.05,
            to=0.95,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            bg="#0B1E30",
            fg="#DDE8F5",
            troughcolor="#173A55",
            highlightthickness=0,
            activebackground="#1E8BD2",
            length=250,
        ).pack(anchor="w", padx=16)

        tk.Label(
            parent,
            text=f"Modelo: {MODEL_PATH.name}",
            bg="#0B1E30",
            fg="#9CB1C7",
            font=("Segoe UI", 9),
            wraplength=260,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=16, pady=(18, 0))

    def _build_main(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg="#0E253B")
        header.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            header,
            text="Painel de Conformidade EPI",
            bg="#0E253B",
            fg="#F4F7FB",
            font=("Segoe UI Semibold", 16),
        ).pack(side=tk.LEFT, padx=14, pady=10)

        self.status_badge = tk.Label(
            header,
            text="AGUARDANDO ANALISE",
            bg="#4A5568",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
        )
        self.status_badge.pack(side=tk.RIGHT, padx=14)

        preview_frame = tk.Frame(parent, bg="#0A1A2A")
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.preview = tk.Label(
            preview_frame,
            bg="#0A1A2A",
            fg="#9AB2C8",
            text="Abra uma imagem ou capture da webcam",
            font=("Segoe UI", 12),
        )
        self.preview.pack(fill=tk.BOTH, expand=True)

        self.countdown_label = tk.Label(
            preview_frame,
            bg="#0A1A2A",
            fg="#F7B32B",
            font=("Segoe UI Black", 36),
            text="",
        )
        self.countdown_label.place(relx=0.5, rely=0.5, anchor="center")

        report_frame = tk.Frame(parent, bg="#0E253B", height=220)
        report_frame.pack(fill=tk.X, pady=(12, 0))
        report_frame.pack_propagate(False)

        tk.Label(
            report_frame,
            text="Relatorio da ultima analise",
            bg="#0E253B",
            fg="#EAF2FA",
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w", padx=12, pady=(10, 6))

        self.report_text = tk.Text(
            report_frame,
            height=8,
            bg="#0B1E30",
            fg="#D8E5F2",
            insertbackground="#D8E5F2",
            relief=tk.FLAT,
            font=("Consolas", 10),
            padx=10,
            pady=8,
        )
        self.report_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.report_text.configure(state=tk.DISABLED)

    def _styled_button(self, parent: tk.Widget, label: str, command, color: str) -> tk.Button:
        return tk.Button(
            parent,
            text=label,
            command=command,
            bg=color,
            fg="white",
            activebackground="#1F2937",
            activeforeground="white",
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=10,
            cursor="hand2",
        )

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")]
        )
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
            img_np = np.array(image)
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao abrir imagem:\n{exc}")
            return

        self.last_source = Path(path).name
        self._start_processing(img_np)

    def capture_webcam(self) -> None:
        if self.countdown_running:
            return

        self.countdown_running = True
        self._set_status("CAPTURANDO...", "#7B4D12")
        self._run_countdown(3)

    def _run_countdown(self, seconds_left: int) -> None:
        if seconds_left > 0:
            self.countdown_label.configure(text=str(seconds_left))
            self.root.after(1000, lambda: self._run_countdown(seconds_left - 1))
            return

        self.countdown_label.configure(text="")
        self._capture_single_frame()

    def _capture_single_frame(self) -> None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            self.countdown_running = False
            self._set_status("ERRO WEBCAM", "#8B1E3F")
            messagebox.showerror("Erro", "Nao foi possivel abrir a webcam.")
            return

        ret, frame = cap.read()
        cap.release()
        self.countdown_running = False

        if not ret:
            self._set_status("ERRO CAPTURA", "#8B1E3F")
            messagebox.showerror("Erro", "Falha ao capturar frame da webcam.")
            return

        self.last_source = "webcam"
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._start_processing(rgb)

    def _start_processing(self, img_rgb: np.ndarray) -> None:
        if self.processing_lock.locked():
            messagebox.showinfo("Aguarde", "Ja existe uma analise em andamento.")
            return

        self.preview.config(text="Processando deteccao...", image="")
        self._set_status("ANALISANDO", "#1A4F7A")

        thread = threading.Thread(target=self._process_image, args=(img_rgb,), daemon=True)
        thread.start()

    def _process_image(self, img_rgb: np.ndarray) -> None:
        with self.processing_lock:
            try:
                annotated, detections, report = run_detection(img_rgb, self.conf.get())
                report.source = self.last_source
                self.current_image = annotated
                self.current_report = report
                self.reports_history.append(report)
                self.root.after(0, lambda: self._render_result(annotated, detections, report))
            except Exception as exc:
                self.root.after(0, lambda: self._handle_error(exc))

    def _render_result(
        self,
        annotated: Image.Image,
        detections: list[DetectionItem],
        report: ComplianceReport,
    ) -> None:
        self._show_pil(annotated)
        color = "#0C8A50" if report.status == "CONFORME" else "#B53152"
        if report.status == "SEM DETECCOES":
            color = "#6B7280"
        self._set_status(report.status, color)
        self._update_report_text(report, detections)

    def _handle_error(self, exc: Exception) -> None:
        self._set_status("ERRO", "#8B1E3F")
        messagebox.showerror("Erro", f"Falha no processamento:\n{exc}")

    def _show_pil(self, pil_img: Image.Image) -> None:
        img = pil_img.copy()
        img.thumbnail((900, 520), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self.preview.image = tk_img
        self.preview.config(image=tk_img, text="")

    def _set_status(self, status: str, color: str) -> None:
        self.status_badge.configure(text=status, bg=color)

    def _update_report_text(self, report: ComplianceReport, detections: list[DetectionItem]) -> None:
        detection_lines = []
        for det in sorted(detections, key=lambda x: x.confidence, reverse=True):
            detection_lines.append(f"- {det.class_name:<30} conf={det.confidence:.2f}")

        if not detection_lines:
            detection_lines.append("- nenhum objeto detectado")

        lines = [
            f"Data/Hora........: {report.timestamp}",
            f"Origem...........: {report.source}",
            f"Status...........: {report.status}",
            f"Score conformidade: {report.compliant_score:.1f}%",
            f"Total deteccoes..: {report.total_detections}",
            f"Nao conformes....: {report.non_compliant_count}",
            "",
            "Resumo:",
            report.summary,
            "",
            "Deteccoes:",
            *detection_lines,
        ]

        self.report_text.configure(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, "\n".join(lines))
        self.report_text.configure(state=tk.DISABLED)

    def save_image(self) -> None:
        if self.current_image is None:
            messagebox.showinfo("Nada para salvar", "Nao ha imagem anotada para salvar.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")],
        )
        if not path:
            return

        try:
            self.current_image.save(path)
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{path}")
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao salvar imagem:\n{exc}")

    def export_report_csv(self) -> None:
        if not self.reports_history:
            messagebox.showinfo("Sem dados", "Nenhum relatorio para exportar.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        "timestamp",
                        "source",
                        "status",
                        "compliant_score",
                        "total_detections",
                        "non_compliant_count",
                        "summary",
                    ]
                )
                for report in self.reports_history:
                    writer.writerow(
                        [
                            report.timestamp,
                            report.source,
                            report.status,
                            f"{report.compliant_score:.2f}",
                            report.total_detections,
                            report.non_compliant_count,
                            report.summary,
                        ]
                    )
            messagebox.showinfo("Sucesso", f"Relatorio exportado em:\n{path}")
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao exportar CSV:\n{exc}")

    def _on_close(self) -> None:
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    YOLOTkApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
