import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, butter, filtfilt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import variation, linregress, gaussian_kde
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para las gráficas
plt.style.use('default')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 1.5

# =============================================================================
# FUNCIONES BASE PARA LECTURA Y PROCESAMIENTO
# =============================================================================

def read_binary_foot_data(filename):
    """
    Lee archivos binarios .let o .rit
    """
    try:
        with open(filename, 'rb') as f:
            raw_data = f.read()
        
        # Probar formatos comunes
        if len(raw_data) % 2 == 0: 
            try:
                data = np.frombuffer(raw_data, dtype=np.int16)
                return data
            except:
                pass
        
        if len(raw_data) % 4 == 0:
            try:
                data = np.frombuffer(raw_data, dtype=np.int32)
                return data
            except:
                pass
        
        # Último recurso
        data = np.frombuffer(raw_data, dtype=np.uint8)
        return data
        
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")
        return None

def eliminar_primeros_segundos(signal_data, segundos_a_eliminar=30, sample_rate=300):
    """
    Elimina los primeros N segundos de la señal
    """
    muestras_a_eliminar = segundos_a_eliminar * sample_rate
    if len(signal_data) > muestras_a_eliminar:
        return signal_data[muestras_a_eliminar:]
    else:
        print(f"Advertencia: Señal demasiado corta para eliminar {segundos_a_eliminar} segundos")
        return signal_data

def aplicar_filtro_cascada(signal_data, sample_rate=300):
    """
    Aplica múltiples filtros en cascada para eliminar ruido persistente
    """
    signal_float = signal_data.astype(np.float64)
    
    # 1. FILTRO PASA-BAJOS AGRESIVO (elimina alta frecuencia)
    nyquist = sample_rate / 2
    b1, a1 = butter(4, 8/nyquist, btype='low')
    etapa1 = filtfilt(b1, a1, signal_float)
    
    # 2. FILTRO MEDIA MÓVIL (suavizado adicional)
    window_size = 50
    window = np.ones(window_size) / window_size
    etapa2 = np.convolve(etapa1, window, mode='same')
    
    # 3. SAVITZKY-GOLAY CON PARÁMETROS MÁS AGRESIVOS
    etapa3 = savgol_filter(etapa2, window_length=51, polyorder=3)
    
    # 4. FILTRO PASA-BAJOS FINAL (pulido)
    b2, a2 = butter(2, 12/nyquist, btype='low')
    señal_filtrada = filtfilt(b2, a2, etapa3)
    
    return señal_filtrada

def normalizar_con_filtro_avanzado(signal_data, target_min=0, target_max=3.5, sample_rate=300):
    """
    Normalización con filtrado avanzado en múltiples etapas
    """
    # Convertir a float
    signal_float = signal_data.astype(np.float64)
    
    # Aplicar filtrado en cascada
    señal_filtrada = aplicar_filtro_cascada(signal_float, sample_rate)
    
    # Normalizar al rango 0-3.5V
    signal_min = np.min(señal_filtrada)
    signal_max = np.max(señal_filtrada)
    
    normalized = (señal_filtrada - signal_min) / (signal_max - signal_min) * (target_max - target_min) + target_min
    
    return normalized

def calcular_derivada_suavizada(signal_voltage, sample_rate=300, window_size=5):
    """
    Calcula la derivada suavizada de la señal para detectar cambios bruscos
    """
    # Calcular derivada simple
    derivative = np.diff(signal_voltage, prepend=signal_voltage[0]) * sample_rate
    
    # Suavizar la derivada para reducir ruido
    window = np.ones(window_size) / window_size
    derivative_smooth = np.convolve(derivative, window, mode='same')
    
    return derivative_smooth

def detectar_zancadas_completo(signal_voltage, sample_rate=300):
    """
    Detecta tanto INICIO como FIN de zancada usando método adaptativo robusto
    """
    print("Aplicando detector completo de zancadas (inicio + fin)...")
    
    # Calcular derivada
    derivative = calcular_derivada_suavizada(signal_voltage, sample_rate)
    
    # Análisis estadístico de la derivada para ajustar umbrales
    der_std = np.std(derivative)
    der_mean = np.mean(derivative)
    der_abs_max = np.max(np.abs(derivative))
    
    print(f"Estadísticas derivada: media={der_mean:.2f}, std={der_std:.2f}, max_abs={der_abs_max:.2f}")
    
    # Estrategias de umbrales para INICIOS (heel strike) y FINES (toe off)
    threshold_strategies = [
        {'start': der_std * 1.5, 'end': -der_std * 1.2, 'name': 'Estrategia 1 (std)'},
        {'start': der_std * 1.2, 'end': -der_std * 1.0, 'name': 'Estrategia 2 (std)'},
        {'start': der_std * 1.0, 'end': -der_std * 0.8, 'name': 'Estrategia 3 (std)'},
        {'start': der_abs_max * 0.4, 'end': -der_abs_max * 0.3, 'name': 'Estrategia 4 (max)'},
        {'start': der_abs_max * 0.3, 'end': -der_abs_max * 0.2, 'name': 'Estrategia 5 (max)'},
        {'start': 15, 'end': -12, 'name': 'Estrategia 6 (fija)'},
        {'start': 10, 'end': -8, 'name': 'Estrategia 7 (fija)'},
        {'start': 8, 'end': -6, 'name': 'Estrategia 8 (fija)'},
        {'start': 5, 'end': -4, 'name': 'Estrategia 9 (fija)'},
    ]
    
    best_detections = 0
    best_starts = []
    best_ends = []
    best_strategy = ""
    
    for strategy in threshold_strategies:
        try:
            # Encontrar INICIOS de zancada (heel strike - picos positivos en derivada)
            starts, start_properties = signal.find_peaks(
                derivative, 
                height=strategy['start'], 
                distance=int(0.3 * sample_rate),  # mín 0.3s entre inicios
                prominence=der_std * 0.3
            )
            
            # Encontrar FINES de zancada (toe off - picos negativos en derivada)
            ends, end_properties = signal.find_peaks(
                -derivative, 
                height=-strategy['end'], 
                distance=int(0.3 * sample_rate),  # mín 0.3s entre fines
                prominence=der_std * 0.3
            )
            
            print(f"  {strategy['name']}: {len(starts)} inicios, {len(ends)} fines")
            
            # Emparejar inicios y fines para formar zancadas completas
            initial_contacts_temp = []
            end_contacts_temp = []
            
            i, j = 0, 0
            while i < len(starts) and j < len(ends):
                current_start = starts[i]
                
                # Buscar el primer fin que ocurra DESPUÉS del inicio actual
                while j < len(ends) and ends[j] <= current_start:
                    j += 1
                
                if j < len(ends):
                    current_end = ends[j]
                    
                    # Verificar que sea una zancada válida (duración razonable)
                    stance_duration = (current_end - current_start) / sample_rate
                    if 0.2 <= stance_duration <= 1.5:  # 200ms a 1.5s
                        initial_contacts_temp.append(current_start)
                        end_contacts_temp.append(current_end)
                        
                        # Buscar siguiente inicio DESPUÉS del fin actual
                        i += 1
                        while i < len(starts) and starts[i] <= current_end:
                            i += 1  # Saltar inicios durante la fase de apoyo
                        
                        j += 1
                    else:
                        # Duración inválida, probar siguiente fin
                        j += 1
                else:
                    break
            
            # Si encontramos más zancadas válidas que la mejor anterior, actualizar
            if len(initial_contacts_temp) > best_detections and len(initial_contacts_temp) >= 2:
                best_detections = len(initial_contacts_temp)
                best_starts = initial_contacts_temp
                best_ends = end_contacts_temp
                best_strategy = strategy['name']
                
        except Exception as e:
            print(f"  {strategy['name']} falló: {e}")
            continue
    
    # Si no encontramos suficientes zancadas con derivada, usar método alternativo
    if best_detections < 3:
        print("Probando método alternativo basado en cambios bruscos en señal...")
        
        # Método alternativo: buscar cambios significativos en la señal original
        signal_diff = np.diff(signal_voltage, prepend=signal_voltage[0])
        threshold_signal = np.std(signal_voltage) * 0.4
        
        # Encontrar cambios positivos significativos (INICIOS - heel strike)
        positive_changes = np.where(signal_diff > threshold_signal)[0]
        # Encontrar cambios negativos significativos (FIN - toe off)
        negative_changes = np.where(signal_diff < -threshold_signal)[0]
        
        # Filtrar por distancia mínima
        min_distance = int(0.4 * sample_rate)
        
        positive_changes_filtered = []
        if len(positive_changes) > 0:
            positive_changes_filtered = [positive_changes[0]]
            for change in positive_changes[1:]:
                if change - positive_changes_filtered[-1] > min_distance:
                    positive_changes_filtered.append(change)
        
        negative_changes_filtered = []
        if len(negative_changes) > 0:
            negative_changes_filtered = [negative_changes[0]]
            for change in negative_changes[1:]:
                if change - negative_changes_filtered[-1] > min_distance:
                    negative_changes_filtered.append(change)
        
        # Emparejar inicios y fines
        initial_contacts_temp = []
        end_contacts_temp = []
        
        i, j = 0, 0
        while i < len(positive_changes_filtered) and j < len(negative_changes_filtered):
            current_start = positive_changes_filtered[i]
            
            # Buscar primer fin DESPUÉS del inicio
            while j < len(negative_changes_filtered) and negative_changes_filtered[j] <= current_start:
                j += 1
            
            if j < len(negative_changes_filtered):
                current_end = negative_changes_filtered[j]
                
                # Verificar duración razonable
                stance_duration = (current_end - current_start) / sample_rate
                if 0.2 <= stance_duration <= 1.5:
                    initial_contacts_temp.append(current_start)
                    end_contacts_temp.append(current_end)
                    
                    i += 1
                    j += 1
                else:
                    j += 1
            else:
                break
        
        if len(initial_contacts_temp) > best_detections:
            best_starts = initial_contacts_temp
            best_ends = end_contacts_temp
            best_strategy = "Método alternativo (señal)"
            best_detections = len(initial_contacts_temp)
        
        print(f"  Método alternativo: {len(positive_changes_filtered)} inicios, {len(negative_changes_filtered)} fines, {len(initial_contacts_temp)} zancadas válidas")
    
    # Asegurar emparejamiento correcto
    min_length = min(len(best_starts), len(best_ends))
    if len(best_starts) != len(best_ends):
        print(f"⚠️  Ajustando emparejamiento: {len(best_starts)} inicios vs {len(best_ends)} fines")
        best_starts = best_starts[:min_length]
        best_ends = best_ends[:min_length]
    
    # Convertir a tiempos
    initial_contacts_times = np.array(best_starts) / sample_rate
    end_contacts_times = np.array(best_ends) / sample_rate
    stance_durations = (np.array(best_ends) - np.array(best_starts)) / sample_rate * 1000
    
    print(f"✅ Detección completada: {len(initial_contacts_times)} zancadas ({best_strategy})")
    
    if len(initial_contacts_times) > 0:
        print(f"   Duración media de apoyo: {np.mean(stance_durations):.1f} ms")
        print(f"   Rango de duraciones: {np.min(stance_durations):.1f} - {np.max(stance_durations):.1f} ms")
    
    return initial_contacts_times, end_contacts_times, stance_durations, derivative

def calcular_intervalos_zancadas(initial_contacts_times):
    """
    Calcula los intervalos entre zancadas consecutivas
    """
    if len(initial_contacts_times) < 2:
        print("⚠️  No hay suficientes zancadas para calcular intervalos")
        return np.array([]), np.array([])
    
    step_intervals = np.diff(initial_contacts_times)  # en segundos
    step_intervals_ms = step_intervals * 1000  # en milisegundos
    
    print(f"✅ Intervalos calculados: {len(step_intervals)} intervalos")
    print(f"   Rango: {np.min(step_intervals_ms):.1f} - {np.max(step_intervals_ms):.1f} ms")
    
    return step_intervals, step_intervals_ms

def filtrar_outliers_mediana(step_intervals, factor=2.5):
    """
    Filtra outliers usando el método basado en mediana y MAD
    """
    if len(step_intervals) == 0:
        print("⚠️  No hay intervalos para filtrar")
        return np.array([]), np.array([])
    
    median = np.median(step_intervals)
    mad = np.median(np.abs(step_intervals - median))  # Median Absolute Deviation
    
    # Calcular límites
    lower_bound = median - factor * mad
    upper_bound = median + factor * mad
    
    # Filtrar outliers
    mask = (step_intervals >= lower_bound) & (step_intervals <= upper_bound)
    filtered_intervals = step_intervals[mask]
    outliers = step_intervals[~mask]
    
    print(f"✅ Filtrado de outliers:")
    print(f"   Total intervalos: {len(step_intervals)}")
    print(f"   Outliers detectados: {len(outliers)}")
    print(f"   Intervalos válidos: {len(filtered_intervals)}")
    
    if len(outliers) > 0:
        print(f"   Valores outliers: {outliers*1000}")
    
    return filtered_intervals, outliers

def calcular_dfa_avanzado(step_intervals, min_window=4, max_window=None):
    """
    Cálculo avanzado del exponente DFA para análisis de fluctuaciones
    """
    if len(step_intervals) < 20:
        print("⚠️  Insuficientes datos para DFA confiable")
        return 0.5, np.array([]), np.array([]), {}
    
    try:
        # 1. Integrar la serie
        y = np.cumsum(step_intervals - np.mean(step_intervals))
        n = len(y)
        
        # 2. Determinar tamaños de ventana
        if max_window is None:
            max_window = n // 4
        
        # Crear ventanas
        window_sizes = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 
                                           num=15, dtype=int))
        window_sizes = window_sizes[window_sizes <= max_window]
        window_sizes = window_sizes[window_sizes >= min_window]
        
        F_n = []
        valid_window_sizes = []
        
        # 3. Calcular fluctuación para cada tamaño de ventana
        for window_size in window_sizes:
            n_segments = n // window_size
            if n_segments < 2:
                continue
                
            F_segment = []
            for i in range(n_segments):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size
                
                if end_idx > n:
                    continue
                    
                segment = y[start_idx:end_idx]
                x = np.arange(len(segment))
                
                # Ajuste lineal
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)
                
                # Fluctuación RMS
                fluctuation = np.sqrt(np.mean((segment - trend) ** 2))
                F_segment.append(fluctuation)
            
            if len(F_segment) > 0:
                F_n.append(np.sqrt(np.mean(np.array(F_segment) ** 2)))
                valid_window_sizes.append(window_size)
        
        if len(F_n) < 3:
            return 0.5, np.array([]), np.array([]), {}
        
        # 4. Ajuste lineal en escala log-log
        log_w = np.log10(valid_window_sizes)
        log_F = np.log10(F_n)
        
        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(log_w, log_F)
        alpha = slope
        
        dfa_metrics = {
            'alpha': alpha,
            'r_cuadrado': r_value ** 2,
            'n_windows': len(valid_window_sizes)
        }
        
        print(f"✅ DFA avanzado calculado: α = {alpha:.3f} (R² = {r_value**2:.3f})")
        
        return alpha, log_w, log_F, dfa_metrics
        
    except Exception as e:
        print(f"❌ Error en DFA avanzado: {e}")
        return 0.5, np.array([]), np.array([]), {}

def calcular_autocorrelacion_avanzada(step_intervals, max_lag=10):
    """
    Calcula la función de autocorrelación con análisis avanzado
    """
    if len(step_intervals) < max_lag * 2:
        max_lag = max(1, len(step_intervals) // 3)
    
    autocorr = []
    lags = range(0, max_lag + 1)
    
    # Autocorrelación para diferentes lags
    for lag in lags:
        if lag >= len(step_intervals):
            break
            
        try:
            if lag == 0:
                corr = 1.0
            else:
                corr = np.corrcoef(step_intervals[:-lag], step_intervals[lag:])[0, 1]
                corr = corr if not np.isnan(corr) else 0
            autocorr.append(corr)
        except:
            autocorr.append(0)
    
    autocorr = np.array(autocorr)
    lags = np.array(lags)
    
    # Métricas de autocorrelación
    decay_lag = None
    for i in range(1, len(autocorr)):
        if autocorr[i] < 0.1:  # Primer lag donde autocorrelación < 0.1
            decay_lag = lags[i]
            break
    
    autocorr_metrics = {
        'autocorr_lag1': autocorr[1] if len(autocorr) > 1 else 0,
        'decay_lag': decay_lag
    }
    
    print(f"✅ Autocorrelación avanzada calculada para {len(autocorr)} lags")
    print(f"   Autocorrelación lag-1: {autocorr_metrics['autocorr_lag1']:.3f}")
    
    return autocorr, lags, autocorr_metrics

def calcular_amplitudes_heel_strikes_avanzado(signal_voltage, initial_contacts_times, sample_rate=300):
    """
    Análisis avanzado de amplitudes de heel strikes
    """
    if len(initial_contacts_times) == 0:
        return np.array([]), {}
    
    amplitudes = []
    time = np.arange(len(signal_voltage)) / sample_rate
    
    for contact_time in initial_contacts_times:
        idx = np.argmin(np.abs(time - contact_time))
        if idx < len(signal_voltage):
            amplitudes.append(signal_voltage[idx])
    
    amplitudes = np.array(amplitudes)
    
    # Métricas de amplitud
    if len(amplitudes) > 0:
        amplitude_metrics = {
            'media': np.mean(amplitudes),
            'desviacion_estandar': np.std(amplitudes),
            'coeficiente_variacion': (np.std(amplitudes) / np.mean(amplitudes) * 100) if np.mean(amplitudes) > 0 else 0,
            'minimo': np.min(amplitudes),
            'maximo': np.max(amplitudes)
        }
    else:
        amplitude_metrics = {}
    
    print(f"✅ Análisis avanzado de amplitudes: {len(amplitudes)} valores")
    return amplitudes, amplitude_metrics

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN (TODAS LAS NECESARIAS)
# =============================================================================

def visualizar_tiempo(signal_voltage_left, signal_voltage_right, sample_rate=300):
    """
    Visualización de ambas señales en el dominio del tiempo
    """
    time_left = np.arange(len(signal_voltage_left)) / sample_rate
    time_right = np.arange(len(signal_voltage_right)) / sample_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Pie izquierdo
    ax1.plot(time_left, signal_voltage_left, 'b-', alpha=0.8, linewidth=1.0)
    ax1.set_ylabel('Amplitud Pie Izquierdo (V)')
    ax1.set_title('COMPARATIVA: Señales Filtradas en el Dominio del Tiempo', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Pie Izquierdo: {len(signal_voltage_left)} muestras', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Pie derecho
    ax2.plot(time_right, signal_voltage_right, 'r-', alpha=0.8, linewidth=1.0)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud Pie Derecho (V)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Pie Derecho: {len(signal_voltage_right)} muestras', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Limitar a una ventana específica para mejor visualización
    if time_left[-1] > 60:
        ax1.set_xlim(30, 40)  # Mostrar entre 30-40 segundos
        ax2.set_xlim(30, 40)
    
    plt.tight_layout()
    plt.show()

def visualizar_zancadas(signal_voltage_left, signal_voltage_right, 
                                  left_initial, right_initial, 
                                  left_end, right_end, sample_rate=300):
    """
    Visualización de detección de zancadas
    """
    time_left = np.arange(len(signal_voltage_left)) / sample_rate
    time_right = np.arange(len(signal_voltage_right)) / sample_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Pie izquierdo con detecciones
    ax1.plot(time_left, signal_voltage_left, 'b-', alpha=0.7, linewidth=1.0, label='Señal filtrada')
    
    if len(left_initial) > 0:
        left_initial_amplitudes = np.interp(left_initial, time_left, signal_voltage_left)
        ax1.scatter(left_initial, left_initial_amplitudes, color='green', s=40, 
                   zorder=5, label='Inicio (heel strike)', marker='^', alpha=0.8)
    
    if len(left_end) > 0:
        left_end_amplitudes = np.interp(left_end, time_left, signal_voltage_left)
        ax1.scatter(left_end, left_end_amplitudes, color='red', s=40, 
                   zorder=5, label='Fin (toe off)', marker='v', alpha=0.8)
    
    # Sombrear fases de apoyo
    for i, (start, end) in enumerate(zip(left_initial, left_end)):
        if i == 0:
            ax1.axvspan(start, end, alpha=0.2, color='green', label='Fase de apoyo')
        else:
            ax1.axvspan(start, end, alpha=0.2, color='green')
    
    ax1.set_ylabel('Amplitud Pie Izquierdo (V)')
    ax1.set_title('Detección de Zancadas - Inicios y Fines', fontweight='bold', pad=20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, f'Pie Izquierdo: {len(left_initial)} zancadas detectadas', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Pie derecho con detecciones
    ax2.plot(time_right, signal_voltage_right, 'r-', alpha=0.7, linewidth=1.0, label='Señal filtrada')
    
    if len(right_initial) > 0:
        right_initial_amplitudes = np.interp(right_initial, time_right, signal_voltage_right)
        ax2.scatter(right_initial, right_initial_amplitudes, color='green', s=40, 
                   zorder=5, label='Inicio (heel strike)', marker='^', alpha=0.8)
    
    if len(right_end) > 0:
        right_end_amplitudes = np.interp(right_end, time_right, signal_voltage_right)
        ax2.scatter(right_end, right_end_amplitudes, color='red', s=40, 
                   zorder=5, label='Fin (toe off)', marker='v', alpha=0.8)
    
    # Sombrear fases de apoyo
    for i, (start, end) in enumerate(zip(right_initial, right_end)):
        if i == 0:
            ax2.axvspan(start, end, alpha=0.2, color='green', label='Fase de apoyo')
        else:
            ax2.axvspan(start, end, alpha=0.2, color='green')
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Amplitud Pie Derecho (V)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f'Pie Derecho: {len(right_initial)} zancadas detectadas', 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Ajustar límites del eje X para mostrar una ventana específica
    if len(time_left) > 0:
        ax1.set_xlim(30, 40)  # Mostrar entre 30-40 segundos
        ax2.set_xlim(30, 40)
    
    plt.tight_layout()
    plt.show()

def visualizar_amplitudes(left_amplitudes, right_amplitudes):
    """
    Visualización de amplitudes de zancadas
    """
    if len(left_amplitudes) == 0 and len(right_amplitudes) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if len(left_amplitudes) > 0:
        left_mean = np.mean(left_amplitudes)
        left_std = np.std(left_amplitudes)
        left_zancadas = np.arange(len(left_amplitudes))
        
        ax1.plot(left_zancadas, left_amplitudes, 'o-', color='blue', markersize=3, 
                linewidth=1.2, alpha=0.7, label='Amplitudes')
        ax1.axhline(y=left_mean, color='red', linestyle='-', linewidth=2, 
                   label=f'Media: {left_mean:.3f}V')
        ax1.axhline(y=left_mean + left_std, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {left_mean + left_std:.3f}V')
        ax1.axhline(y=left_mean - left_std, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {left_mean - left_std:.3f}V')
        
        ax1.set_xlabel('Zancada')
        ax1.set_ylabel('Amplitud (V)')
        ax1.set_title(f'Pie Izquierdo - Amplitudes (σ = {left_std:.3f}V)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = left_std * 1.5
        ax1.set_ylim(left_mean - y_margin, left_mean + y_margin)
    
    # Pie derecho
    if len(right_amplitudes) > 0:
        right_mean = np.mean(right_amplitudes)
        right_std = np.std(right_amplitudes)
        right_zancadas = np.arange(len(right_amplitudes))
        
        ax2.plot(right_zancadas, right_amplitudes, 'o-', color='red', markersize=3, 
                linewidth=1.2, alpha=0.7, label='Amplitudes')
        ax2.axhline(y=right_mean, color='darkred', linestyle='-', linewidth=2, 
                   label=f'Media: {right_mean:.3f}V')
        ax2.axhline(y=right_mean + right_std, color='darkred', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {right_mean + right_std:.3f}V')
        ax2.axhline(y=right_mean - right_std, color='darkred', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {right_mean - right_std:.3f}V')
        
        ax2.set_xlabel('Zancada')
        ax2.set_ylabel('Amplitud (V)')
        ax2.set_title(f'Pie Derecho - Amplitudes (σ = {right_std:.3f}V)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = right_std * 1.5
        ax2.set_ylim(right_mean - y_margin, right_mean + y_margin)
    
    plt.suptitle('Amplitudes de Zancadas', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualizar_duraciones(left_intervals, right_intervals):
    """
    Visualización de duraciones de zancadas
    """
    if len(left_intervals) == 0 and len(right_intervals) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if len(left_intervals) > 0:
        left_mean = np.mean(left_intervals)
        left_std = np.std(left_intervals)
        left_zancadas = np.arange(len(left_intervals))
        
        ax1.plot(left_zancadas, left_intervals, 'o-', color='green', markersize=3, 
                linewidth=1.2, alpha=0.7, label='Duración')
        ax1.axhline(y=left_mean, color='blue', linestyle='-', linewidth=2, 
                   label=f'Media: {left_mean:.3f}s')
        ax1.axhline(y=left_mean + left_std, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {left_mean + left_std:.3f}s')
        ax1.axhline(y=left_mean - left_std, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {left_mean - left_std:.3f}s')
        
        ax1.set_xlabel('Zancada')
        ax1.set_ylabel('Duración (s)')
        ax1.set_title(f'Pie Izquierdo - Duración (σ = {left_std:.3f}s)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = left_std * 1.5
        ax1.set_ylim(max(0.1, left_mean - y_margin), left_mean + y_margin)
    
    # Pie derecho
    if len(right_intervals) > 0:
        right_mean = np.mean(right_intervals)
        right_std = np.std(right_intervals)
        right_zancadas = np.arange(len(right_intervals))
        
        ax2.plot(right_zancadas, right_intervals, 'o-', color='orange', markersize=3, 
                linewidth=1.2, alpha=0.7, label='Duración')
        ax2.axhline(y=right_mean, color='purple', linestyle='-', linewidth=2, 
                   label=f'Media: {right_mean:.3f}s')
        ax2.axhline(y=right_mean + right_std, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {right_mean + right_std:.3f}s')
        ax2.axhline(y=right_mean - right_std, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {right_mean - right_std:.3f}s')
        
        ax2.set_xlabel('Zancada')
        ax2.set_ylabel('Duración (s)')
        ax2.set_title(f'Pie Derecho - Duración (σ = {right_std:.3f}s)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = right_std * 1.5
        ax2.set_ylim(max(0.1, right_mean - y_margin), right_mean + y_margin)
    
    plt.suptitle('Duración de Zancadas', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualizar_dfa(left_dfa_data, right_dfa_data):
    """
    Visualización del análisis DFA
    """
    has_left = len(left_dfa_data.get('log_w', [])) > 0
    has_right = len(right_dfa_data.get('log_w', [])) > 0
    
    if not has_left and not has_right:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if has_left:
        left_log_w = left_dfa_data['log_w']
        left_log_F = left_dfa_data['log_F']
        left_alpha = left_dfa_data['alpha']
        
        ax1.plot(left_log_w, left_log_F, 'o-', color='darkorange', markersize=5, 
                linewidth=1.5, alpha=0.7, label='Datos DFA')
        
        if len(left_log_w) > 1:
            x_fit = np.linspace(np.min(left_log_w), np.max(left_log_w), 100)
            intercept = np.mean(left_log_F) - left_alpha * np.mean(left_log_w)
            y_fit = left_alpha * x_fit + intercept
            ax1.plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f'Ajuste (α = {left_alpha:.3f})')
        
        ax1.set_xlabel('log(Window Size)')
        ax1.set_ylabel('log(Fluctuation)')
        ax1.set_title(f'Pie Izquierdo - DFA (α = {left_alpha:.3f})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Pie derecho
    if has_right:
        right_log_w = right_dfa_data['log_w']
        right_log_F = right_dfa_data['log_F']
        right_alpha = right_dfa_data['alpha']
        
        ax2.plot(right_log_w, right_log_F, 'o-', color='darkcyan', markersize=5, 
                linewidth=1.5, alpha=0.7, label='Datos DFA')
        
        if len(right_log_w) > 1:
            x_fit = np.linspace(np.min(right_log_w), np.max(right_log_w), 100)
            intercept = np.mean(right_log_F) - right_alpha * np.mean(right_log_w)
            y_fit = right_alpha * x_fit + intercept
            ax2.plot(x_fit, y_fit, 'r--', linewidth=2, 
                    label=f'Ajuste (α = {right_alpha:.3f})')
        
        ax2.set_xlabel('log(Window Size)')
        ax2.set_ylabel('log(Fluctuation)')
        ax2.set_title(f'Pie Derecho - DFA (α = {right_alpha:.3f})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis DFA (Detrended Fluctuation Analysis)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualizar_autocorrelacion(left_autocorr_data, right_autocorr_data):
    has_left = len(left_autocorr_data[0]) > 0
    has_right = len(right_autocorr_data[0]) > 0
    
    if not has_left and not has_right:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if has_left:
        left_autocorr = left_autocorr_data[0]
        left_lags = left_autocorr_data[1]
        
        # Gráfico de puntos con líneas
        ax1.plot(left_lags, left_autocorr, 'o-', color='teal', markersize=6, 
                linewidth=2, alpha=0.8, markerfacecolor='white', markeredgecolor='teal',
                markeredgewidth=2)
        
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Marcar punto de decaimiento
        decay_lag = None
        for i, (lag, corr) in enumerate(zip(left_lags, left_autocorr)):
            if i > 0 and abs(corr) < 0.1:
                decay_lag = lag
                break
        
        if decay_lag is not None:
            ax1.axvline(x=decay_lag, color='red', linestyle=':', alpha=0.7,
                       label=f'Decaimiento: {decay_lag} strides')
            ax1.legend()
        
        ax1.set_xlabel('Lag (strides)')
        ax1.set_ylabel('Autocorrelación')
        ax1.set_title('Pie Izquierdo - Autocorrelación', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
    
    # Pie derecho
    if has_right:
        right_autocorr = right_autocorr_data[0]
        right_lags = right_autocorr_data[1]
        
        # Gráfico de puntos con líneas
        ax2.plot(right_lags, right_autocorr, 'o-', color='brown', markersize=6, 
                linewidth=2, alpha=0.8, markerfacecolor='white', markeredgecolor='brown',
                markeredgewidth=2)
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Marcar punto de decaimiento
        decay_lag = None
        for i, (lag, corr) in enumerate(zip(right_lags, right_autocorr)):
            if i > 0 and abs(corr) < 0.1:
                decay_lag = lag
                break
        
        if decay_lag is not None:
            ax2.axvline(x=decay_lag, color='red', linestyle=':', alpha=0.7,
                       label=f'Decaimiento: {decay_lag} strides')
            ax2.legend()
        
        ax2.set_xlabel('Lag (strides)')
        ax2.set_ylabel('Autocorrelación')
        ax2.set_title('Pie Derecho - Autocorrelación', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.1, 1.1)
    
    plt.suptitle('Función de Autocorrelación', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
def visualizar_distribuciones_superpuestas(left_intervals, right_intervals):
    """
    Visualización de distribuciones de intervalos SUPERPUESTAS
    """
    if len(left_intervals) == 0 and len(right_intervals) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determinar rangos comunes para los bins
    all_intervals = []
    if len(left_intervals) > 0:
        all_intervals.extend(left_intervals)
    if len(right_intervals) > 0:
        all_intervals.extend(right_intervals)
    
    if len(all_intervals) == 0:
        return
    
    # Crear bins comunes para ambas distribuciones
    min_val = np.min(all_intervals)
    max_val = np.max(all_intervals)
    bins = np.linspace(min_val, max_val, 20)
    
    # Pie izquierdo - histograma
    if len(left_intervals) > 0:
        left_color = 'blue'
        n_left, bins_left, patches_left = ax.hist(left_intervals, bins=bins, alpha=0.6, 
                                                color=left_color, edgecolor='darkblue', 
                                                linewidth=1.2, density=True, 
                                                label=f'Pie Izquierdo (n={len(left_intervals)})')
        
        # Línea de densidad KDE para pie izquierdo
        kde_left = gaussian_kde(left_intervals)
        x_range_left = np.linspace(min_val, max_val, 200)
        ax.plot(x_range_left, kde_left(x_range_left), color=left_color, 
               linewidth=3, label='KDE Pie Izquierdo')
        
        # Línea vertical para la media del pie izquierdo
        left_mean = np.mean(left_intervals)
        ax.axvline(x=left_mean, color=left_color, linestyle='--', linewidth=2, 
                  alpha=0.8, label=f'Media Izq: {left_mean:.3f}s')
    
    # Pie derecho - histograma
    if len(right_intervals) > 0:
        right_color = 'red'
        n_right, bins_right, patches_right = ax.hist(right_intervals, bins=bins, alpha=0.6, 
                                                   color=right_color, edgecolor='darkred', 
                                                   linewidth=1.2, density=True, 
                                                   label=f'Pie Derecho (n={len(right_intervals)})')
        
        # Línea de densidad KDE para pie derecho
        kde_right = gaussian_kde(right_intervals)
        x_range_right = np.linspace(min_val, max_val, 200)
        ax.plot(x_range_right, kde_right(x_range_right), color=right_color, 
               linewidth=3, label='KDE Pie Derecho')
        
        # Línea vertical para la media del pie derecho
        right_mean = np.mean(right_intervals)
        ax.axvline(x=right_mean, color=right_color, linestyle='--', linewidth=2, 
                  alpha=0.8, label=f'Media Der: {right_mean:.3f}s')
    
    # Configuración del gráfico
    ax.set_xlabel('Intervalo de Zancada (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Densidad de Probabilidad', fontsize=12, fontweight='bold')
    ax.set_title('Intervalos de Zancada - Ambos Pies', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Leyenda y grid
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Estadísticas en el gráfico
    stats_text = ""
    if len(left_intervals) > 0:
        left_std = np.std(left_intervals)
        left_cv = (left_std / left_mean * 100) if left_mean > 0 else 0
        stats_text += f"Pie Izquierdo:\n  μ={left_mean:.3f}s, σ={left_std:.3f}s\n  CV={left_cv:.1f}%\n\n"
    
    if len(right_intervals) > 0:
        right_std = np.std(right_intervals)
        right_cv = (right_std / right_mean * 100) if right_mean > 0 else 0
        stats_text += f"Pie Derecho:\n  μ={right_mean:.3f}s, σ={right_std:.3f}s\n  CV={right_cv:.1f}%"
    
    # Añadir cuadro de texto con estadísticas
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Ajustar límites
    ax.set_xlim(min_val - 0.05, max_val + 0.05)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# NUEVAS FUNCIONES DE VISUALIZACIÓN ESPECÍFICAS
# =============================================================================

def visualizar_señal_normalizada_derivada(signal_voltage, initial_contacts, end_contacts, 
                                        derivative, sample_rate=300, pie_nombre="Izquierdo"):
    """
    Visualización de señal normalizada y su derivada (similar a Control1_2.png y Control1_3.png)
    """
    time = np.arange(len(signal_voltage)) / sample_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Gráfico superior: Señal normalizada
    ax1.plot(time, signal_voltage, 'b-', linewidth=1.5, alpha=0.8, label='Señal normalizada')
    
    # Marcar inicios y fines de zancada
    if len(initial_contacts) > 0:
        initial_amplitudes = np.interp(initial_contacts, time, signal_voltage)
        ax1.scatter(initial_contacts, initial_amplitudes, color='green', s=60, 
                   zorder=5, label='Inicio zancada (heel strike)', marker='^')
    
    if len(end_contacts) > 0:
        end_amplitudes = np.interp(end_contacts, time, signal_voltage)
        ax1.scatter(end_contacts, end_amplitudes, color='red', s=60, 
                   zorder=5, label='Fin zancada (toe off)', marker='v')
    
    # Sombrear fases de apoyo
    for start, end in zip(initial_contacts, end_contacts):
        ax1.axvspan(start, end, alpha=0.2, color='gray', label='Fase de apoyo' if start == initial_contacts[0] else "")
    
    ax1.set_ylabel('Voltaje (V)')
    ax1.set_title(f'Pie {pie_nombre} - Señal Normalizada y Derivada - Señal Normalizada (0-3.5V)', 
                 fontweight='bold', pad=20)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 3.6)
    
    # Gráfico inferior: Derivada de la señal
    time_derivative = time[:len(derivative)]
    ax2.plot(time_derivative, derivative, 'purple', linewidth=1.5, alpha=0.8, label='Derivada de la señal')
    
    # Estadísticas de la derivada
    der_mean = np.mean(derivative)
    der_std = np.std(derivative)
    zancadas_detectadas = len(initial_contacts)
    
    # Marcar umbrales en derivada
    if len(initial_contacts) > 0:
        # Encontrar valores de derivada en los puntos de inicio
        start_indices = (initial_contacts * sample_rate).astype(int)
        start_indices = start_indices[start_indices < len(derivative)]
        if len(start_indices) > 0:
            start_derivative_values = derivative[start_indices]
            umbral_inicio = np.percentile(start_derivative_values, 75) if len(start_derivative_values) > 0 else 10
            
            ax2.axhline(y=umbral_inicio, color='green', linestyle='--', alpha=0.7, 
                       label=f'Umbral inicio ({umbral_inicio:.1f})')
    
    if len(end_contacts) > 0:
        # Encontrar valores de derivada en los puntos de fin
        end_indices = (end_contacts * sample_rate).astype(int)
        end_indices = end_indices[end_indices < len(derivative)]
        if len(end_indices) > 0:
            end_derivative_values = derivative[end_indices]
            umbral_fin = np.percentile(end_derivative_values, 25) if len(end_derivative_values) > 0 else -8
            
            ax2.axhline(y=umbral_fin, color='red', linestyle='--', alpha=0.7, 
                       label=f'Umbral fin ({umbral_fin:.1f})')
    
    # Marcar puntos detectados en derivada
    if len(initial_contacts) > 0:
        start_indices = (initial_contacts * sample_rate).astype(int)
        start_indices = start_indices[start_indices < len(derivative)]
        ax2.scatter(initial_contacts, derivative[start_indices], color='green', s=40, 
                   zorder=5, label='Inicio detectado', marker='^')
    
    if len(end_contacts) > 0:
        end_indices = (end_contacts * sample_rate).astype(int)
        end_indices = end_indices[end_indices < len(derivative)]
        ax2.scatter(end_contacts, derivative[end_indices], color='red', s=40, 
                   zorder=5, label='Fin detectado', marker='v')
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Derivada (V/s)')
    ax2.set_title('Derivada de la Señal - Detección de Cambios Bruscos', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Añadir estadísticas en el gráfico
    stats_text = f'Estadísticas derivadas:\nMedia: {der_mean:.2f} V/s\nStd: {der_std:.2f} V/s\nZancadas detectadas: {zancadas_detectadas}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Limitar a una ventana de tiempo específica para mejor visualización
    if len(time) > 0:
        ax1.set_xlim(30, 40)  # Mostrar entre 30-40 segundos como ejemplo
        ax2.set_xlim(30, 40)
    
    plt.tight_layout()
    plt.show()

def visualizar_duracion_zancadas_detallado(left_intervals, right_intervals):
    """
    Visualización detallada de duración de zancadas (similar a Control1_7.png)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Pie izquierdo
    if len(left_intervals) > 0:
        left_mean = np.mean(left_intervals)
        left_std = np.std(left_intervals)
        left_zancadas = np.arange(len(left_intervals))
        
        ax1.plot(left_zancadas, left_intervals, 'o-', color='blue', markersize=4, 
                linewidth=1.2, alpha=0.8)
        
        # Líneas de media y desviación estándar
        ax1.axhline(y=left_mean, color='red', linestyle='-', linewidth=2, 
                   label=f'Media: {left_mean:.3f}s')
        ax1.axhline(y=left_mean + left_std, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {left_mean + left_std:.3f}s')
        ax1.axhline(y=left_mean - left_std, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {left_mean - left_std:.3f}s')
        
        # Rellenar área entre ±1σ
        ax1.fill_between(left_zancadas, left_mean - left_std, left_mean + left_std, 
                        alpha=0.2, color='red')
        
        ax1.set_xlabel('Zancada')
        ax1.set_ylabel('Duración (s)')
        ax1.set_title(f'Pie Izquierdo - Duración (σ = {left_std:.3f}s)', fontweight='bold', pad=20)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = left_std * 1.2
        ax1.set_ylim(max(0.1, left_mean - y_margin), left_mean + y_margin)
        ax1.set_xlim(0, len(left_intervals))
    
    # Pie derecho
    if len(right_intervals) > 0:
        right_mean = np.mean(right_intervals)
        right_std = np.std(right_intervals)
        right_zancadas = np.arange(len(right_intervals))
        
        ax2.plot(right_zancadas, right_intervals, 'o-', color='red', markersize=4, 
                linewidth=1.2, alpha=0.8)
        
        # Líneas de media y desviación estándar
        ax2.axhline(y=right_mean, color='blue', linestyle='-', linewidth=2, 
                   label=f'Media: {right_mean:.3f}s')
        ax2.axhline(y=right_mean + right_std, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {right_mean + right_std:.3f}s')
        ax2.axhline(y=right_mean - right_std, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {right_mean - right_std:.3f}s')
        
        # Rellenar área entre ±1σ
        ax2.fill_between(right_zancadas, right_mean - right_std, right_mean + right_std, 
                        alpha=0.2, color='blue')
        
        ax2.set_xlabel('Zancada')
        ax2.set_ylabel('Duración (s)')
        ax2.set_title(f'Pie Derecho - Duración (σ = {right_std:.3f}s)', fontweight='bold', pad=20)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Ajustar límites
        y_margin = right_std * 1.2
        ax2.set_ylim(max(0.1, right_mean - y_margin), right_mean + y_margin)
        ax2.set_xlim(0, len(right_intervals))
    
    plt.suptitle('Duración de Zancadas', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

def visualizar_dfa_detallado(left_dfa_data, right_dfa_data):
    """
    Visualización detallada del análisis DFA (similar a Control1_8.png)
    """
    has_left = len(left_dfa_data.get('log_w', [])) > 0
    has_right = len(right_dfa_data.get('log_w', [])) > 0
    
    if not has_left and not has_right:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if has_left:
        left_log_w = left_dfa_data['log_w']
        left_log_F = left_dfa_data['log_F']
        left_alpha = left_dfa_data['alpha']
        
        ax1.scatter(left_log_w, left_log_F, color='darkorange', s=50, 
                   alpha=0.7, label='Datos DFA')
        
        if len(left_log_w) > 1:
            # Ajuste lineal
            slope, intercept, r_value, p_value, std_err = linregress(left_log_w, left_log_F)
            x_fit = np.linspace(np.min(left_log_w), np.max(left_log_w), 100)
            y_fit = slope * x_fit + intercept
            
            ax1.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Ajuste (α = {left_alpha:.3f})')
        
        ax1.set_xlabel('log(Window Size)')
        ax1.set_ylabel('log(Fluctuation)')
        ax1.set_title(f'Pie Izquierdo - DFA (α = {left_alpha:.3f})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Pie derecho
    if has_right:
        right_log_w = right_dfa_data['log_w']
        right_log_F = right_dfa_data['log_F']
        right_alpha = right_dfa_data['alpha']
        
        ax2.scatter(right_log_w, right_log_F, color='darkcyan', s=50, 
                   alpha=0.7, label='Datos DFA')
        
        if len(right_log_w) > 1:
            # Ajuste lineal
            slope, intercept, r_value, p_value, std_err = linregress(right_log_w, right_log_F)
            x_fit = np.linspace(np.min(right_log_w), np.max(right_log_w), 100)
            y_fit = slope * x_fit + intercept
            
            ax2.plot(x_fit, y_fit, 'r-', linewidth=2, 
                    label=f'Ajuste (α = {right_alpha:.3f})')
        
        ax2.set_xlabel('log(Window Size)')
        ax2.set_ylabel('log(Fluctuation)')
        ax2.set_title(f'Pie Derecho - DFA (α = {right_alpha:.3f})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis DFA (Detrended Fluctuation Analysis)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualizar_autocorrelacion_detallado(left_autocorr_data, right_autocorr_data):
  
    has_left = len(left_autocorr_data[0]) > 0
    has_right = len(right_autocorr_data[0]) > 0
    
    if not has_left and not has_right:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo
    if has_left:
        left_autocorr = left_autocorr_data[0]
        left_lags = left_autocorr_data[1]
        
        # Crear gráfico de puntos para autocorrelación
        ax1.scatter(left_lags, left_autocorr, color='skyblue', s=80, 
                   alpha=0.8, edgecolor='navy', linewidth=1.5, zorder=5)
        
        # Conectar puntos con líneas
        ax1.plot(left_lags, left_autocorr, 'o-', color='skyblue', 
                markersize=6, linewidth=2, alpha=0.7)
        
        # Añadir línea en y=0
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
        
        # Líneas de referencia en ±0.1
        ax1.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax1.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Encontrar punto de decaimiento
        decay_lag_left = None
        for i, (lag, corr) in enumerate(zip(left_lags, left_autocorr)):
            if i > 0 and abs(corr) < 0.1:
                decay_lag_left = lag
                break
        
        if decay_lag_left is not None:
            ax1.axvline(x=decay_lag_left, color='red', linestyle=':', alpha=0.8, linewidth=2,
                       label=f'Decaimiento: {decay_lag_left} strides')
            ax1.legend()
        
        ax1.set_xlabel('Lag (strides)')
        ax1.set_ylabel('Autocorrelación')
        ax1.set_title('Pie Izquierdo - Autocorrelación', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlim(-0.5, max(left_lags) + 0.5)
        
        # Añadir valores numéricos en los puntos
        for i, (lag, corr) in enumerate(zip(left_lags, left_autocorr)):
            ax1.annotate(f'{corr:.2f}', (lag, corr), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    # Pie derecho
    if has_right:
        right_autocorr = right_autocorr_data[0]
        right_lags = right_autocorr_data[1]
        
        # Crear gráfico de puntos para autocorrelación
        ax2.scatter(right_lags, right_autocorr, color='lightcoral', s=80, 
                   alpha=0.8, edgecolor='darkred', linewidth=1.5, zorder=5)
        
        # Conectar puntos con líneas
        ax2.plot(right_lags, right_autocorr, 'o-', color='lightcoral', 
                markersize=6, linewidth=2, alpha=0.7)
        
        # Añadir línea en y=0
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5, linewidth=1)
        
        # Líneas de referencia en ±0.1
        ax2.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.axhline(y=-0.1, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
        
        # Encontrar punto de decaimiento
        decay_lag_right = None
        for i, (lag, corr) in enumerate(zip(right_lags, right_autocorr)):
            if i > 0 and abs(corr) < 0.1:
                decay_lag_right = lag
                break
        
        if decay_lag_right is not None:
            ax2.axvline(x=decay_lag_right, color='red', linestyle=':', alpha=0.8, linewidth=2,
                       label=f'Decaimiento: {decay_lag_right} strides')
            ax2.legend()
        
        ax2.set_xlabel('Lag (strides)')
        ax2.set_ylabel('Autocorrelación')
        ax2.set_title('Pie Derecho - Autocorrelación', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlim(-0.5, max(right_lags) + 0.5)
        
        # Añadir valores numéricos en los puntos
        for i, (lag, corr) in enumerate(zip(right_lags, right_autocorr)):
            ax2.annotate(f'{corr:.2f}', (lag, corr), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    plt.suptitle('Función de Autocorrelación', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()
def visualizar_zancada_vs_amplitud(left_amplitudes, right_amplitudes, left_intervals, right_intervals):
    """
    Visualización de zancada vs amplitud (volts) - NUEVA GRÁFICA SOLICITADA
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie izquierdo: zancada vs amplitud
    if len(left_amplitudes) > 0 and len(left_intervals) > 0:
        min_len = min(len(left_amplitudes), len(left_intervals))
        zancadas = np.arange(min_len)
        
        ax1.plot(zancadas, left_amplitudes[:min_len], 'o-', color='blue', 
                markersize=4, linewidth=1.2, alpha=0.8)
        
        ax1.set_xlabel('Número de Zancada')
        ax1.set_ylabel('Amplitud (V)')
        ax1.set_title('Pie Izquierdo - Amplitud vs Zancada', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Calcular estadísticas
        mean_amp = np.mean(left_amplitudes[:min_len])
        std_amp = np.std(left_amplitudes[:min_len])
        
        ax1.axhline(y=mean_amp, color='red', linestyle='-', linewidth=2,
                   label=f'Media: {mean_amp:.2f}V')
        ax1.axhline(y=mean_amp + std_amp, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {mean_amp + std_amp:.2f}V')
        ax1.axhline(y=mean_amp - std_amp, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {mean_amp - std_amp:.2f}V')
        ax1.legend()
    
    # Pie derecho: zancada vs amplitud
    if len(right_amplitudes) > 0 and len(right_intervals) > 0:
        min_len = min(len(right_amplitudes), len(right_intervals))
        zancadas = np.arange(min_len)
        
        ax2.plot(zancadas, right_amplitudes[:min_len], 'o-', color='red', 
                markersize=4, linewidth=1.2, alpha=0.8)
        
        ax2.set_xlabel('Número de Zancada')
        ax2.set_ylabel('Amplitud (V)')
        ax2.set_title('Pie Derecho - Amplitud vs Zancada', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Calcular estadísticas
        mean_amp = np.mean(right_amplitudes[:min_len])
        std_amp = np.std(right_amplitudes[:min_len])
        
        ax2.axhline(y=mean_amp, color='blue', linestyle='-', linewidth=2,
                   label=f'Media: {mean_amp:.2f}V')
        ax2.axhline(y=mean_amp + std_amp, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'+1σ: {mean_amp + std_amp:.2f}V')
        ax2.axhline(y=mean_amp - std_amp, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'-1σ: {mean_amp - std_amp:.2f}V')
        ax2.legend()
    
    plt.suptitle('Relación Zancada vs Amplitud (Volts)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.show()

def visualizar_señal_original_no_filtrada(left_raw, right_raw, sample_rate=300):
    """
    Visualización de señal original no filtrada (similar a Control1_1.png)
    """
    time_left = np.arange(len(left_raw)) / sample_rate
    time_right = np.arange(len(right_raw)) / sample_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Pie izquierdo - señal original
    ax1.plot(time_left, left_raw, 'b-', alpha=0.8, linewidth=1.0)
    
    # Estadísticas para pie izquierdo
    left_mean = np.mean(left_raw)
    left_std = np.std(left_raw)
    left_range = f"{np.min(left_raw):.1f}-{np.max(left_raw):.1f}V"
    
    stats_text_left = f"Estadísticas Pre-Izquierdo:\nMedia: {left_mean:.2f} V\nDesv. Estándar: {left_std:.2f} V\nRango: {left_range}\nMuestras: {len(left_raw)}"
    
    ax1.text(0.02, 0.98, stats_text_left, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax1.set_ylabel('Voltaje (V)')
    ax1.set_title('SEÑAL ORIGINAL NO FILTRADA - Pie Izquierdo', fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # Pie derecho - señal original
    ax2.plot(time_right, right_raw, 'r-', alpha=0.8, linewidth=1.0)
    
    # Estadísticas para pie derecho
    right_mean = np.mean(right_raw)
    right_std = np.std(right_raw)
    right_range = f"{np.min(right_raw):.1f}-{np.max(right_raw):.1f}V"
    
    stats_text_right = f"Estadísticas Pre-Derecho:\nMedia: {right_mean:.2f} V\nDesv. Estándar: {right_std:.2f} V\nRango: {right_range}\nMuestras: {len(right_raw)}"
    
    ax2.text(0.02, 0.98, stats_text_right, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Voltaje (V)')
    ax2.set_title('SEÑAL ORIGINAL NO FILTRADA - Pie Derecho', fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    # Limitar a una ventana de tiempo razonable
    max_display_time = min(60, time_left[-1], time_right[-1])  # Mostrar primeros 60 segundos o menos
    ax1.set_xlim(0, max_display_time)
    ax2.set_xlim(0, max_display_time)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FUNCIÓN ACTUALIZADA DE VISUALIZACIONES COMPLETAS
# =============================================================================

def crear_visualizaciones_completas(signal_voltage_left, signal_voltage_right,
                                   left_initial, right_initial,
                                   left_end, right_end,
                                   left_intervals, right_intervals,
                                   left_dfa_data, right_dfa_data,
                                   left_autocorr_data, right_autocorr_data,
                                   left_amplitudes, right_amplitudes,
                                   left_derivative, right_derivative,
                                   left_raw, right_raw,
                                   sample_rate=300):
    """
    Crea TODAS las visualizaciones incluyendo las nuevas específicas
    """
    print("\n📊 CREANDO VISUALIZACIONES COMPLETAS...")
    
    # 1. Señal original no filtrada
    visualizar_señal_original_no_filtrada(left_raw, right_raw, sample_rate)
    
    # 2. Señales en dominio del tiempo filtradas
    visualizar_tiempo(signal_voltage_left, signal_voltage_right, sample_rate)
    
    # 3. Señal normalizada y derivada para cada pie
    visualizar_señal_normalizada_derivada(signal_voltage_left, left_initial, left_end, 
                                         left_derivative, sample_rate, "Izquierdo")
    visualizar_señal_normalizada_derivada(signal_voltage_right, right_initial, right_end, 
                                         right_derivative, sample_rate, "Derecho")
    
    # 4. Detección de zancadas
    visualizar_zancadas(signal_voltage_left, signal_voltage_right,
                       left_initial, right_initial,
                       left_end, right_end, sample_rate)
    
    # 5. Duración de zancadas (detallado)
    visualizar_duracion_zancadas_detallado(left_intervals, right_intervals)
    
    # 6. Amplitudes de zancadas
    visualizar_amplitudes(left_amplitudes, right_amplitudes)
    
    # 7. Zancada vs Amplitud (NUEVA)
    visualizar_zancada_vs_amplitud(left_amplitudes, right_amplitudes, left_intervals, right_intervals)
    
    # 8. Análisis DFA (detallado)
    visualizar_dfa_detallado(left_dfa_data, right_dfa_data)
    
    # 9. Autocorrelación (detallado)
    visualizar_autocorrelacion_detallado(left_autocorr_data, right_autocorr_data)
    
    # 10. Distribución de intervalos SUPERPUESTA
    visualizar_distribuciones_superpuestas(left_intervals, right_intervals)

# =============================================================================
# FUNCIÓN PRINCIPAL COMPLETA
# =============================================================================

def main_comparativo_completo():
    """
    Función principal que ejecuta el análisis completo con todas las visualizaciones
    """
    print("=== ANÁLISIS COMPARATIVO COMPLETO DE MARCHA - AMBOS PIES ===")
    print("Cargando y procesando datos...")
    
    # 1. Cargar datos (mantener los datos originales para visualización)
    left_raw = read_binary_foot_data("control3.let")
    right_raw = read_binary_foot_data("control3.rit")
    
    if left_raw is None or right_raw is None:
        print("Error: No se pudieron leer los archivos")
        print("Asegúrate de que los archivos estén en el directorio correcto")
        return
    
    # 2. Preprocesamiento (mantener una copia de los datos sin procesar para visualización)
    segundos_a_eliminar = 30
    left_sin_30s = eliminar_primeros_segundos(left_raw, segundos_a_eliminar)
    right_sin_30s = eliminar_primeros_segundos(right_raw, segundos_a_eliminar)
    
    print(f"✅ Datos cargados:")
    print(f"   Pie izquierdo: {len(left_sin_30s)} muestras")
    print(f"   Pie derecho: {len(right_sin_30s)} muestras")
    
    left_avanzado = normalizar_con_filtro_avanzado(left_sin_30s)
    right_avanzado = normalizar_con_filtro_avanzado(right_sin_30s)
    
    # 3. Detección COMPLETA de zancadas (INICIO + FIN)
    print("\n🔍 DETECTANDO ZANCADAS COMPLETAS (INICIO + FIN)...")
    left_initial, left_end, left_stance_durations, left_derivative = detectar_zancadas_completo(left_avanzado)
    right_initial, right_end, right_stance_durations, right_derivative = detectar_zancadas_completo(right_avanzado)
    
    if len(left_initial) < 2 and len(right_initial) < 2:
        print("\n❌ ANÁLISIS NO PUEDE CONTINUAR")
        print("No se detectaron suficientes zancadas para el análisis.")
        return
    
    # 4. Cálculo de intervalos
    print("\n📏 CALCULANDO INTERVALOS...")
    left_intervals, left_intervals_ms = calcular_intervalos_zancadas(left_initial)
    right_intervals, right_intervals_ms = calcular_intervalos_zancadas(right_initial)
    
    # 5. Filtrado de outliers
    print("\n🗑️  FILTRANDO OUTLIERS...")
    left_intervals_filt, left_outliers = filtrar_outliers_mediana(left_intervals)
    right_intervals_filt, right_outliers = filtrar_outliers_mediana(right_intervals)
    
    left_intervals_analisis = left_intervals_filt if len(left_intervals_filt) > 0 else left_intervals
    right_intervals_analisis = right_intervals_filt if len(right_intervals_filt) > 0 else right_intervals
    
    # 6. Cálculo de métricas avanzadas
    print("\n📊 CALCULANDO MÉTRICAS AVANZADAS...")
    
    # DFA avanzado
    left_alpha, left_log_w, left_log_F, left_dfa_metrics = calcular_dfa_avanzado(left_intervals_analisis)
    right_alpha, right_log_w, right_log_F, right_dfa_metrics = calcular_dfa_avanzado(right_intervals_analisis)
    
    # Autocorrelación avanzada
    left_autocorr, left_autocorr_lags, left_autocorr_metrics = calcular_autocorrelacion_avanzada(left_intervals_analisis)
    right_autocorr, right_autocorr_lags, right_autocorr_metrics = calcular_autocorrelacion_avanzada(right_intervals_analisis)
    
    # Amplitudes
    left_amplitudes, left_amplitude_metrics = calcular_amplitudes_heel_strikes_avanzado(left_avanzado, left_initial)
    right_amplitudes, right_amplitude_metrics = calcular_amplitudes_heel_strikes_avanzado(right_avanzado, right_initial)
    
    # 7. Crear TODAS las visualizaciones
    crear_visualizaciones_completas(
        left_avanzado, right_avanzado,
        left_initial, right_initial,
        left_end, right_end,
        left_intervals_analisis, right_intervals_analisis,
        {'log_w': left_log_w, 'log_F': left_log_F, 'alpha': left_alpha},
        {'log_w': right_log_w, 'log_F': right_log_F, 'alpha': right_alpha},
        (left_autocorr, left_autocorr_lags),
        (right_autocorr, right_autocorr_lags),
        left_amplitudes, right_amplitudes,
        left_derivative, right_derivative,
        left_sin_30s, right_sin_30s  # Datos sin normalizar para visualización original
    )
    
    # 8. Presentar resumen numérico 
    print("\n📋 RESUMEN NUMÉRICO COMPLETO:")
    print("=" * 60)
    
    if len(left_intervals_analisis) > 0:
        left_mean = np.mean(left_intervals_analisis)
        left_std = np.std(left_intervals_analisis)
        left_cv = (left_std / left_mean * 100) if left_mean > 0 else 0
        print(f"PIE IZQUIERDO:")
        print(f"  • Intervalo medio: {left_mean:.3f}s")
        print(f"  • Desviación estándar: {left_std:.3f}s")
        print(f"  • Coeficiente de variación: {left_cv:.1f}%")
        print(f"  • Exponente DFA: {left_alpha:.3f}")
        print(f"  • Autocorrelación lag-1: {left_autocorr_metrics.get('autocorr_lag1', 0):.3f}")
        if len(left_stance_durations) > 0:
            print(f"  • Duración media de apoyo: {np.mean(left_stance_durations):.1f} ms")
        print()
    
    if len(right_intervals_analisis) > 0:
        right_mean = np.mean(right_intervals_analisis)
        right_std = np.std(right_intervals_analisis)
        right_cv = (right_std / right_mean * 100) if right_mean > 0 else 0
        print(f"PIE DERECHO:")
        print(f"  • Intervalo medio: {right_mean:.3f}s")
        print(f"  • Desviación estándar: {right_std:.3f}s")
        print(f"  • Coeficiente de variación: {right_cv:.1f}%")
        print(f"  • Exponente DFA: {right_alpha:.3f}")
        print(f"  • Autocorrelación lag-1: {right_autocorr_metrics.get('autocorr_lag1', 0):.3f}")
        if len(right_stance_durations) > 0:
            print(f"  • Duración media de apoyo: {np.mean(right_stance_durations):.1f} ms")
        print()
    
    # Comparación directa
    if len(left_intervals_analisis) > 0 and len(right_intervals_analisis) > 0:
        print("COMPARACIÓN DIRECTA:")
        diff_mean = abs(left_mean - right_mean)
        diff_std = abs(left_std - right_std)
        diff_alpha = abs(left_alpha - right_alpha)
        
        print(f"  • Diferencia en intervalo medio: {diff_mean:.3f}s")
        print(f"  • Diferencia en variabilidad: {diff_std:.3f}s")
        print(f"  • Diferencia en DFA: {diff_alpha:.3f}")
        
        if diff_mean < 0.05 and diff_std < 0.05:
            print("  ✅ Los pies muestran patrones similares")
        else:
            print("  ⚠️  Se observan diferencias entre los pies")
    
    print("\n✅ ANÁLISIS COMPLETADO EXITOSAMENTE")

if __name__ == "__main__":
    main_comparativo_completo()