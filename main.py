"""
=============================================================================
 CALCULADORA DE VALORES CRÍTICOS DE PROBABILIDAD
 Distribuciones: Z (Normal), t-Student, Chi-Cuadrado, F de Fisher
=============================================================================
 Librerías: flet, numpy, matplotlib
 Sin scipy ni librerías estadísticas de alto nivel.
 Todos los cálculos implementados con aproximaciones numéricas.
=============================================================================
"""

import flet as ft
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64
import io
import math

# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 1 ─ FUNCIONES MATEMÁTICAS BASE
# ─────────────────────────────────────────────────────────────────────────────

def ln_gamma(z: float) -> float:
    """
    Log-Gamma usando la aproximación de Lanczos (g=7, 9 coeficientes).
    Exactitud > 15 dígitos para Re(z) > 0.5.
    """
    g = 7
    c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    if z < 0.5:
        return math.log(math.pi / math.sin(math.pi * z)) - ln_gamma(1.0 - z)
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return (0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x))


def gamma_func(z: float) -> float:
    """Función Gamma completa."""
    return math.exp(ln_gamma(z))


# ─── Función de error (erf) por serie de Taylor ──────────────────────────────

def erf_series(x: float) -> float:
    """
    erf(x) mediante serie de Taylor:
      erf(x) = (2/√π) Σ [(-1)^n * x^(2n+1)] / [n! * (2n+1)]
    Convergencia rápida para |x| < 3.7; para valores mayores se usa expansión asintótica.
    """
    if abs(x) > 6.0:
        return math.copysign(1.0, x)
    term = x
    total = x
    x2 = x * x
    for n in range(1, 200):
        term *= -x2 / n
        delta = term / (2 * n + 1)
        total += delta
        if abs(delta) < 1e-15 * abs(total):
            break
    return (2.0 / math.sqrt(math.pi)) * total


def erfc(x: float) -> float:
    return 1.0 - erf_series(x)


# ─── CDF Normal estándar ─────────────────────────────────────────────────────

def normal_cdf(x: float) -> float:
    """
    Φ(x) = 0.5 * [1 + erf(x / √2)]
    """
    return 0.5 * (1.0 + erf_series(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


# ─── CDF t-Student ───────────────────────────────────────────────────────────

def beta_regularized_I(x: float, a: float, b: float) -> float:
    """
    I_x(a,b): función beta regularizada incompleta.
    Se calcula via fracción continua (método de Lentz modificado),
    que converge para x < (a+1)/(a+b+2).
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x fuera de [0,1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    lbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    ln_front = math.log(x) * a + math.log(1.0 - x) * b - lbeta - math.log(a)

    # Usar simetría si x > (a+1)/(a+b+2) para mejor convergencia
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - beta_regularized_I(1.0 - x, b, a)

    # Fracción continua de Lentz
    FPMIN = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        # Paso par
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        # Paso impar
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break

    return math.exp(ln_front) * h


def t_cdf(t: float, df: float) -> float:
    """
    CDF de la t-Student con 'df' grados de libertad.
    Usa la relación con la función beta incompleta regularizada:
      P(T ≤ t) = 1 - 0.5 * I_{df/(df+t²)}(df/2, 0.5)   si t ≥ 0
    """
    x = df / (df + t * t)
    p_tail = 0.5 * beta_regularized_I(x, df / 2.0, 0.5)
    if t >= 0:
        return 1.0 - p_tail
    else:
        return p_tail


def t_pdf(t: float, df: float) -> float:
    """PDF de t-Student."""
    coef = math.exp(ln_gamma((df + 1) / 2) - ln_gamma(df / 2))
    coef /= math.sqrt(df * math.pi)
    return coef * (1 + t * t / df) ** (-(df + 1) / 2)


# ─── CDF Chi-Cuadrado ────────────────────────────────────────────────────────

def gamma_incomplete_lower(s: float, x: float) -> float:
    """
    Función gamma incompleta inferior regularizada P(s, x) = γ(s,x)/Γ(s).
    Usa expansión en serie para x < s+1, fracción continua para x ≥ s+1.
    """
    if x < 0:
        raise ValueError("x debe ser ≥ 0")
    if x == 0:
        return 0.0

    if x < s + 1.0:
        # Serie de Taylor
        ap = s
        delta = 1.0 / s
        total = delta
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + s * math.log(x) - ln_gamma(s))
    else:
        # Fracción continua (Legendre)
        FPMIN = 1e-300
        b = x + 1.0 - s
        c = 1.0 / FPMIN
        d = 1.0 / b
        h = d
        for i in range(1, 500):
            an = -i * (i - s)
            b += 2.0
            d = an * d + b
            if abs(d) < FPMIN:
                d = FPMIN
            c = b + an / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        return 1.0 - math.exp(-x + s * math.log(x) - ln_gamma(s)) * h


def chi2_cdf(x: float, df: float) -> float:
    """CDF de Chi-cuadrado: P(χ² ≤ x | df) = γ(df/2, x/2) / Γ(df/2)"""
    if x <= 0:
        return 0.0
    return gamma_incomplete_lower(df / 2.0, x / 2.0)


def chi2_pdf(x: float, df: float) -> float:
    """PDF de Chi-cuadrado."""
    if x <= 0:
        return 0.0
    k = df / 2.0
    return math.exp((k - 1) * math.log(x) - x / 2.0 - k * math.log(2) - ln_gamma(k))


# ─── CDF F de Fisher ─────────────────────────────────────────────────────────

def f_cdf(x: float, d1: float, d2: float) -> float:
    """
    CDF de la distribución F.
    Usa la relación:  P(F ≤ x) = I_{d1*x/(d1*x + d2)}(d1/2, d2/2)
    """
    if x <= 0:
        return 0.0
    t = d1 * x / (d1 * x + d2)
    return beta_regularized_I(t, d1 / 2.0, d2 / 2.0)


def f_pdf(x: float, d1: float, d2: float) -> float:
    """PDF de la distribución F."""
    if x <= 0:
        return 0.0
    try:
        ln_num = (d1 / 2) * math.log(d1) + (d2 / 2) * math.log(d2)
        ln_num += (d1 / 2 - 1) * math.log(x)
        ln_den = ln_gamma(d1 / 2) + ln_gamma(d2 / 2) - ln_gamma((d1 + d2) / 2)
        ln_den += ((d1 + d2) / 2) * math.log(d1 * x + d2)
        return math.exp(ln_num - ln_den)
    except (ValueError, OverflowError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 2 ─ NEWTON-RAPHSON PARA INVERSIÓN DE CDF
# ─────────────────────────────────────────────────────────────────────────────

def find_critical_value(cdf_func, pdf_func, p_target: float,
                        x0: float = 1.0, tol: float = 1e-10,
                        max_iter: int = 200) -> float:
    """
    Método de Newton-Raphson para resolver CDF(x) = p_target.
    Iteración:  x_{n+1} = x_n - [CDF(x_n) - p] / PDF(x_n)
    """
    x = x0
    for _ in range(max_iter):
        fx = cdf_func(x) - p_target
        dfx = pdf_func(x)
        if dfx == 0.0:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def z_critical(alpha: float, two_tailed: bool = True) -> float:
    """Valor crítico Z."""
    p = 1.0 - alpha / 2 if two_tailed else 1.0 - alpha
    return find_critical_value(normal_cdf, normal_pdf, p, x0=1.645)


def t_critical(alpha: float, df: int, two_tailed: bool = True) -> float:
    """Valor crítico t-Student."""
    p = 1.0 - alpha / 2 if two_tailed else 1.0 - alpha
    cdf = lambda x: t_cdf(x, df)
    pdf = lambda x: t_pdf(x, df)
    return find_critical_value(cdf, pdf, p, x0=z_critical(alpha, two_tailed))


def chi2_critical(alpha: float, df: int) -> float:
    """Valor crítico Chi-cuadrado (cola derecha)."""
    p = 1.0 - alpha
    cdf = lambda x: chi2_cdf(x, df)
    pdf = lambda x: chi2_pdf(x, df)
    x0 = max(df * (1 - 2 / (9 * df) + 1.645 * math.sqrt(2 / (9 * df))) ** 3, 1.0)
    return find_critical_value(cdf, pdf, p, x0=x0)


def f_critical(alpha: float, df1: int, df2: int) -> float:
    """Valor crítico F de Fisher (cola derecha)."""
    p = 1.0 - alpha
    cdf = lambda x: f_cdf(x, df1, df2)
    pdf = lambda x: f_pdf(x, df1, df2)
    # Aproximación inicial
    x0 = max((df2 / (df2 - 2)) * 1.5, 1.0) if df2 > 2 else 2.0
    return find_critical_value(cdf, pdf, p, x0=x0)


# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 3 ─ GENERACIÓN DE GRÁFICOS CON MATPLOTLIB
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg": "#1A1A2E",
    "card": "#16213E",
    "primary": "#0F3460",
    "accent": "#E94560",
    "text": "#EAEAEA",
    "shade_reject": "#E94560",
    "shade_accept": "#0F3460",
    "curve": "#00D4FF",
}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def plot_normal(alpha: float, two_tailed: bool, cv: float) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    x = np.linspace(-4, 4, 500)
    y = np.array([normal_pdf(xi) for xi in x])
    ax.plot(x, y, color=COLORS["curve"], lw=2)

    if two_tailed:
        # Zona de rechazo bilateral
        x_right = x[x >= cv]
        ax.fill_between(x_right, [normal_pdf(xi) for xi in x_right],
                        alpha=0.6, color=COLORS["shade_reject"], label=f"α/2={alpha/2:.4f}")
        x_left = x[x <= -cv]
        ax.fill_between(x_left, [normal_pdf(xi) for xi in x_left],
                        alpha=0.6, color=COLORS["shade_reject"])
        ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.axvline(-cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.set_title(f"Normal Z  |  α={alpha}  (bilateral)\nz_c = ±{cv:.4f}",
                     color=COLORS["text"], fontsize=10)
    else:
        x_right = x[x >= cv]
        ax.fill_between(x_right, [normal_pdf(xi) for xi in x_right],
                        alpha=0.6, color=COLORS["shade_reject"], label=f"α={alpha:.4f}")
        ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.set_title(f"Normal Z  |  α={alpha}  (unilateral)\nz_c = {cv:.4f}",
                     color=COLORS["text"], fontsize=10)

    _style_axes(ax)
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def plot_t(alpha: float, df: int, two_tailed: bool, cv: float) -> str:
    lim = max(cv * 1.5, 4.0)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    x = np.linspace(-lim, lim, 600)
    y = np.array([t_pdf(xi, df) for xi in x])
    ax.plot(x, y, color=COLORS["curve"], lw=2)

    if two_tailed:
        x_r = x[x >= cv];  x_l = x[x <= -cv]
        ax.fill_between(x_r, [t_pdf(xi, df) for xi in x_r],
                        alpha=0.6, color=COLORS["shade_reject"])
        ax.fill_between(x_l, [t_pdf(xi, df) for xi in x_l],
                        alpha=0.6, color=COLORS["shade_reject"])
        ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.axvline(-cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.set_title(f"t-Student  |  df={df}  α={alpha}  (bilateral)\nt_c = ±{cv:.4f}",
                     color=COLORS["text"], fontsize=10)
    else:
        x_r = x[x >= cv]
        ax.fill_between(x_r, [t_pdf(xi, df) for xi in x_r],
                        alpha=0.6, color=COLORS["shade_reject"])
        ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
        ax.set_title(f"t-Student  |  df={df}  α={alpha}  (unilateral)\nt_c = {cv:.4f}",
                     color=COLORS["text"], fontsize=10)

    _style_axes(ax)
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def plot_chi2(alpha: float, df: int, cv: float) -> str:
    mode = max(df - 2, 0.1)
    lim = max(cv * 1.5, df + 4 * math.sqrt(2 * df))
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    x = np.linspace(0.01, lim, 600)
    y = np.array([chi2_pdf(xi, df) for xi in x])
    ax.plot(x, y, color=COLORS["curve"], lw=2)

    x_r = x[x >= cv]
    ax.fill_between(x_r, [chi2_pdf(xi, df) for xi in x_r],
                    alpha=0.6, color=COLORS["shade_reject"])
    ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
    ax.set_title(f"Chi-Cuadrado  |  df={df}  α={alpha}\nχ²_c = {cv:.4f}",
                 color=COLORS["text"], fontsize=10)

    _style_axes(ax)
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def plot_f(alpha: float, df1: int, df2: int, cv: float) -> str:
    lim = max(cv * 1.6, 5.0)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["card"])

    x = np.linspace(0.01, lim, 600)
    y = np.array([f_pdf(xi, df1, df2) for xi in x])
    ax.plot(x, y, color=COLORS["curve"], lw=2)

    x_r = x[x >= cv]
    ax.fill_between(x_r, [f_pdf(xi, df1, df2) for xi in x_r],
                    alpha=0.6, color=COLORS["shade_reject"])
    ax.axvline(cv, color=COLORS["accent"], lw=1.5, linestyle="--")
    ax.set_title(f"F de Fisher  |  df1={df1}  df2={df2}  α={alpha}\nF_c = {cv:.4f}",
                 color=COLORS["text"], fontsize=10)

    _style_axes(ax)
    img = _fig_to_base64(fig)
    plt.close(fig)
    return img


def _style_axes(ax):
    """Aplica estilo oscuro consistente a los ejes."""
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.yaxis.label.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text"])
    ax.grid(True, linestyle="--", alpha=0.2, color="#555577")


# ─────────────────────────────────────────────────────────────────────────────
#  BLOQUE 4 ─ INTERFAZ FLET
# ─────────────────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title = "Valores Críticos de Probabilidad"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = COLORS["bg"]
    page.padding = 0
    page.scroll = "auto"

    # ── Paleta de colores Flet ───────────────────────────────────────────────
    C_BG      = COLORS["bg"]
    C_CARD    = COLORS["card"]
    C_PRIMARY = COLORS["primary"]
    C_ACCENT  = COLORS["accent"]
    C_TEXT    = COLORS["text"]

    def card(content, padding=16):
        return ft.Container(
            content=content,
            bgcolor=C_CARD,
            border_radius=12,
            padding=padding,
            margin=ft.margin.only(bottom=10),
        )

    def section_title(text):
        return ft.Text(text, size=13, weight="bold", color=C_ACCENT)

    def result_text_ref():
        return ft.Ref[ft.Text]()

    def plot_ref():
        return ft.Ref[ft.Image]()

    # ── Shared state ─────────────────────────────────────────────────────────
    result_ref  = ft.Ref[ft.Text]()
    graph_ref   = ft.Ref[ft.Image]()
    error_ref   = ft.Ref[ft.Text]()

    def show_error(msg):
        error_ref.current.value = msg
        error_ref.current.visible = True
        error_ref.current.update()

    def clear_error():
        error_ref.current.value = ""
        error_ref.current.visible = False
        error_ref.current.update()

    def show_result(cv_text, img_b64):
        result_ref.current.value = cv_text
        result_ref.current.visible = True
        result_ref.current.update()
        graph_ref.current.src = ""
        graph_ref.current.src_base64 = img_b64
        graph_ref.current.visible = True
        graph_ref.current.update()

    # ─────────────────────────────────────────────────────────────────────────
    #  PESTAÑA Z – Normal Estándar
    # ─────────────────────────────────────────────────────────────────────────
    z_alpha  = ft.TextField(
        label="Nivel de significancia α",
        value="0.05",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    z_tails  = ft.Dropdown(
        label="Tipo de prueba",
        options=[ft.dropdown.Option("bilateral", "Bilateral (dos colas)"),
                 ft.dropdown.Option("unilateral", "Unilateral (una cola)")],
        value="bilateral",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )

    def calc_z(e):
        clear_error()
        try:
            alpha = float(z_alpha.value)
            assert 0 < alpha < 1, "α debe estar en (0,1)"
            two   = z_tails.value == "bilateral"
            cv    = z_critical(alpha, two)
            img   = plot_normal(alpha, two, cv)
            label = f"Valor crítico  Z = ±{cv:.4f}" if two else f"Valor crítico  Z = {cv:.4f}"
            show_result(label, img)
        except Exception as ex:
            show_error(str(ex))

    tab_z = ft.Column([
        card(ft.Column([section_title("Distribución Normal (Z)"), z_alpha, z_tails])),
        ft.ElevatedButton("Calcular", on_click=calc_z,
                          bgcolor=C_ACCENT, color=C_TEXT,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
    ], spacing=6)

    # ─────────────────────────────────────────────────────────────────────────
    #  PESTAÑA t-Student
    # ─────────────────────────────────────────────────────────────────────────
    t_alpha = ft.TextField(
        label="Nivel de significancia α", value="0.05",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    t_df = ft.TextField(
        label="Grados de libertad (df)", value="10",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    t_tails = ft.Dropdown(
        label="Tipo de prueba",
        options=[ft.dropdown.Option("bilateral", "Bilateral (dos colas)"),
                 ft.dropdown.Option("unilateral", "Unilateral (una cola)")],
        value="bilateral",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )

    def calc_t(e):
        clear_error()
        try:
            alpha = float(t_alpha.value)
            df    = int(t_df.value)
            assert 0 < alpha < 1, "α debe estar en (0,1)"
            assert df >= 1, "df debe ser ≥ 1"
            two   = t_tails.value == "bilateral"
            cv    = t_critical(alpha, df, two)
            img   = plot_t(alpha, df, two, cv)
            label = f"Valor crítico  t = ±{cv:.4f}" if two else f"Valor crítico  t = {cv:.4f}"
            show_result(label, img)
        except Exception as ex:
            show_error(str(ex))

    tab_t = ft.Column([
        card(ft.Column([section_title("Distribución t-Student"), t_alpha, t_df, t_tails])),
        ft.ElevatedButton("Calcular", on_click=calc_t,
                          bgcolor=C_ACCENT, color=C_TEXT,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
    ], spacing=6)

    # ─────────────────────────────────────────────────────────────────────────
    #  PESTAÑA Chi-Cuadrado
    # ─────────────────────────────────────────────────────────────────────────
    chi_alpha = ft.TextField(
        label="Nivel de significancia α", value="0.05",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    chi_df = ft.TextField(
        label="Grados de libertad (df)", value="5",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )

    def calc_chi2(e):
        clear_error()
        try:
            alpha = float(chi_alpha.value)
            df    = int(chi_df.value)
            assert 0 < alpha < 1, "α debe estar en (0,1)"
            assert df >= 1, "df debe ser ≥ 1"
            cv    = chi2_critical(alpha, df)
            img   = plot_chi2(alpha, df, cv)
            show_result(f"Valor crítico  χ² = {cv:.4f}", img)
        except Exception as ex:
            show_error(str(ex))

    tab_chi2 = ft.Column([
        card(ft.Column([section_title("Distribución Chi-Cuadrado (χ²)"), chi_alpha, chi_df])),
        ft.ElevatedButton("Calcular", on_click=calc_chi2,
                          bgcolor=C_ACCENT, color=C_TEXT,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
    ], spacing=6)

    # ─────────────────────────────────────────────────────────────────────────
    #  PESTAÑA F de Fisher
    # ─────────────────────────────────────────────────────────────────────────
    f_alpha = ft.TextField(
        label="Nivel de significancia α", value="0.05",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    f_df1 = ft.TextField(
        label="df1 (numerador)", value="3",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )
    f_df2 = ft.TextField(
        label="df2 (denominador)", value="20",
        keyboard_type="number",
        bgcolor=C_PRIMARY, color=C_TEXT, label_style=ft.TextStyle(color="#AAAACC"),
        border_radius=8, border_color="#334466",
    )

    def calc_f(e):
        clear_error()
        try:
            alpha = float(f_alpha.value)
            df1   = int(f_df1.value)
            df2   = int(f_df2.value)
            assert 0 < alpha < 1, "α debe estar en (0,1)"
            assert df1 >= 1 and df2 >= 1, "df deben ser ≥ 1"
            cv    = f_critical(alpha, df1, df2)
            img   = plot_f(alpha, df1, df2, cv)
            show_result(f"Valor crítico  F = {cv:.4f}", img)
        except Exception as ex:
            show_error(str(ex))

    tab_f = ft.Column([
        card(ft.Column([section_title("Distribución F de Fisher"), f_alpha, f_df1, f_df2])),
        ft.ElevatedButton("Calcular", on_click=calc_f,
                          bgcolor=C_ACCENT, color=C_TEXT,
                          style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))),
    ], spacing=6)

    # ─────────────────────────────────────────────────────────────────────────
    #  RESULTADO Y GRÁFICO (compartidos por todas las pestañas)
    # ─────────────────────────────────────────────────────────────────────────
    result_box = ft.Text(
        ref=result_ref,
        value="",
        visible=False,
        size=17,
        weight="bold",
        color=C_ACCENT,
        text_align="center",
    )
    graph_box = ft.Image(
        ref=graph_ref,
        src="https://via.placeholder.com/1x1",
        visible=False,
        fit="contain",
        border_radius=10,
    )
    error_box = ft.Text(
        ref=error_ref,
        value="",
        visible=False,
        color="#FF6B6B",
        size=12,
    )

    result_section = card(
        ft.Column([result_box, error_box, graph_box], spacing=8,
                  horizontal_alignment="center"),
        padding=12,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  NAVEGACIÓN PERSONALIZADA (reemplaza ft.Tabs incompatible)
    # ─────────────────────────────────────────────────────────────────────────
    TABS     = ["Z", "t", "χ²", "F"]
    tab_views = [tab_z, tab_t, tab_chi2, tab_f]
    current_tab = {"index": 0}

    # Contenedor del contenido activo
    content_area = ft.Container(
        content=tab_views[0],
        padding=10,
        expand=True,
    )

    # Referencias a los botones de navegación
    nav_btn_refs = [ft.Ref[ft.Container]() for _ in TABS]

    def make_nav_btn(label, idx):
        is_active = idx == 0
        return ft.Container(
            ref=nav_btn_refs[idx],
            content=ft.Text(
                label,
                size=15,
                weight="bold",
                color=C_ACCENT if is_active else C_TEXT,
                text_align="center",
            ),
            bgcolor=C_CARD if is_active else C_PRIMARY,
            border_radius=8,
            padding=ft.padding.symmetric(horizontal=14, vertical=8),
            border=ft.border.all(2, C_ACCENT) if is_active else ft.border.all(1, "#334466"),
            on_click=lambda e, i=idx: switch_tab(i),
            expand=True,
        )

    def switch_tab(idx):
        current_tab["index"] = idx
        # Actualizar estilo de botones
        for i, ref in enumerate(nav_btn_refs):
            is_active = i == idx
            ref.current.bgcolor = C_CARD if is_active else C_PRIMARY
            ref.current.border = ft.border.all(2, C_ACCENT) if is_active else ft.border.all(1, "#334466")
            ref.current.content.color = C_ACCENT if is_active else C_TEXT
            ref.current.update()
        # Cambiar contenido
        content_area.content = tab_views[idx]
        content_area.update()
        # Limpiar resultado al cambiar pestaña
        result_ref.current.visible = False
        result_ref.current.update()
        graph_ref.current.visible = False
        graph_ref.current.update()
        error_ref.current.visible = False
        error_ref.current.update()

    nav_bar = ft.Row(
        controls=[make_nav_btn(label, i) for i, label in enumerate(TABS)],
        spacing=6,
    )

    # ─────────────────────────────────────────────────────────────────────────
    #  HEADER
    # ─────────────────────────────────────────────────────────────────────────
    header = ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.CALCULATE_ROUNDED, color=C_ACCENT, size=28),
                ft.Text("Valores Críticos", size=20, weight="bold", color=C_TEXT),
            ], alignment="center"),
            ft.Text("Probabilidad & Estadística", size=11, color="#778899",
                    text_align="center"),
        ], horizontal_alignment="center", spacing=2),
        bgcolor=C_PRIMARY,
        padding=ft.padding.symmetric(vertical=14),
        border_radius=ft.border_radius.only(bottom_left=16, bottom_right=16),
        margin=ft.margin.only(bottom=8),
    )

    page.add(
        header,
        ft.Container(
            content=ft.Column([
                nav_bar,
                content_area,
                result_section,
            ], spacing=10),
            padding=ft.padding.symmetric(horizontal=12),
        ),
    )


ft.app(target=main)
