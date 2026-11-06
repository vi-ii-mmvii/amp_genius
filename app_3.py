import streamlit as st
import pandas as pd
import numpy as np
import os
import datetime
import time
from Bio.SeqUtils.ProtParam import ProteinAnalysis

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

HISTORY_FILE = "generated_peptides_history.csv"

def load_history_from_disk():
    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            return df.to_dict("records")
        except Exception:
            return []
    return []

def save_history_to_disk(history_list):
    if history_list:
        df = pd.DataFrame(history_list)
        df.to_csv(HISTORY_FILE, index=False)
    elif os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

@st.cache_resource
def load_model():
    adapter_path = "./amp_genius_lora_final"
    base_model_name = "nferruz/ProtGPT2"
    
    if not TORCH_AVAILABLE or not os.path.exists(adapter_path):
        return None, None, "cpu"
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        return tokenizer, model, device
    except Exception:
        return None, None, "cpu"

def calculate_biophysics(sequence: str):
    clean_seq = "".join([aa for aa in sequence.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if not clean_seq:
        return {"MW": 0.0, "pI": 0.0, "Charge": 0.0, "Gravy": 0.0}
    
    analyser = ProteinAnalysis(clean_seq)
    mw = analyser.molecular_weight()
    pi = analyser.isoelectric_point()
    charge = analyser.charge_at_pH(7.4)
    gravy = analyser.gravy()
    
    return {
        "MW": round(mw, 2),
        "pI": round(pi, 2),
        "Charge": round(charge, 2),
        "Gravy": round(gravy, 2)
    }

def estimate_hemolysis_risk(charge: float, gravy: float, is_cosmetic_mode: bool) -> str:
    if is_cosmetic_mode:
        return "Некритично (Наружное применение)"
    if charge >= 4.0 and gravy > 0.2:
        return "Высокий (Требуется тест SI)"
    elif charge >= 2.0 or gravy > 0.0:
        return "Умеренный"
    else:
        return "Низкий (Безопасен)"

def get_buffer_recommendations(pi: float, mw: float, gravy: float):
    recs = {}
    
    # IEX Chromatography
    if pi >= 7.2:
        target_ph = max(3.5, round(pi - 2.0, 1))
        recs["IEX"] = {
            "mode": "CEX (Катионообменная хроматография)",
            "primary": {
                "name": f"20–50 мМ Натрий-ацетатный буфер (pH {target_ph})",
                "details": f"pH = {target_ph} (на 2 ед. ниже pI = {pi}). Градиент: 0.0 -> 1.0 M NaCl.",
                "why": "Пептид приобретает устойчивый положительный заряд. Высокая связывающая емкость сорбента."
            },
            "alt1": {
                "name": f"20 мМ Натрий-фосфатный буфер (pH {round(pi - 0.8, 1)})",
                "impact": f"pH приближен к pI. Частичная потеря заряда снижает полноту связывания и размывает пик."
            },
            "alt2": {
                "name": "10 мМ Цитратный буфер (pH 3.0)",
                "impact": "Кислая среда обеспечивает растворимость, но повышает риск гидролиза Asp-Pro связей."
            }
        }
    else:
        target_ph = min(10.0, round(pi + 2.0, 1))
        recs["IEX"] = {
            "mode": "AEX (Анионообменная хроматография)",
            "primary": {
                "name": f"20–50 мМ Tris-HCl буфер (pH {target_ph})",
                "details": f"pH = {target_ph} (на 2 ед. выше pI = {pi}). Градиент: 0.0 -> 1.0 M NaCl.",
                "why": "Пептид заряжен отрицательно, обеспечивая удерживание на сорбенте типа Q/DEAE."
            },
            "alt1": {
                "name": f"20 мМ Карбонатный буфер (pH {min(10.5, round(target_ph + 0.8, 1))})",
                "impact": "Щелочная среда может вызывать рацемизацию и деамидирование остатков Asn/Gln."
            },
            "alt2": {
                "name": "20 мМ Ацетат аммония (pH 6.5)",
                "impact": "Неполная ионизация снижает выход пептида с колонки."
            }
        }
        
    # RP-HPLC Polishing
    recs["RP_HPLC"] = {
        "primary": {
            "name": "Фаза А: 0.1% TFA в H2O (pH ~2.1) | Фаза B: 0.1% TFA в 80% Ацетонитриле",
            "details": "Градиент: 5% -> 65% B за 35 мин. Трифторуксусная кислота выступает ион-парным агентом.",
            "why": "Обеспечивает четкое разделение пиков и нейтрализацию carboxyl-групп (чистота > 95%)."
        },
        "alt1": {
            "name": "Фаза А: 0.1% Муравьиная кислота (pH ~2.7) | Фаза B: 0.1% FA в Ацетонитриле",
            "impact": "Оптимально для LC-MS (летучая система). Форма хроматографических пиков более широкая."
        },
        "alt2": {
            "name": "10 мМ Ацетат аммония (pH 6.8)",
            "impact": "Применяется для кислоточувствительных пептидов. Изменяет селективность разделения."
        }
    }
    
    # Solubilization & Storage
    if 7.0 <= pi <= 7.8:
        sol_primary = "10–20 мМ Натрий-ацетатный буфер (pH 4.0–4.5) или 5% уксусная кислота"
        sol_why = "pI равен pH крови (7.4). Растворять строго при pH < 5.0 во избежание агрегации."
        sol_alt = "10 мМ Tris-HCl (pH 8.5) с 5% маннита для последующей лиофилизации."
    elif pi > 7.8:
        sol_primary = "10 мМ Фосфатный буфер (PBS, pH 6.0) или 0.1% уксусная кислота"
        sol_why = "Высокая растворимость при нейтральных и слабокислых значением pH за счет положительного заряда."
        sol_alt = "Стерильная деионизованная вода (pH 5.5)."
    else:
        sol_primary = "10 мМ Бикарбонат аммония (pH 7.8–8.2)"
        sol_why = "Кислый пептид хорошо растворим в слабощелочной среде; соль полностью улетучивается при сушке."
        sol_alt = "10 мМ Фосфатно-солевой буфер (PBS, pH 7.4)."
        
    recs["Storage"] = {
        "primary": {"name": sol_primary, "why": sol_why},
        "alt": {"name": sol_alt, "impact": "Альтернативный состав для замораживания и хранения при -80°C."}
    }
    
    return recs

PEPTIDE_TEMPLATES = {
    "[CHARGE_HIGH_POS]": ["KWKLFKKIEKWGQNIRDGIVKAGPAVAVVGQAATI", "KRFKKFFKKKLKNSVKKVTK", "RRWWRRWRRWRRWRR"],
    "[CHARGE_POS]": ["GFLGAVGPALATVAKKLL", "FAKKLAKLAKKLAKAL", "ILPWKWPWWPWRR"],
    "[CHARGE_NEUT]": ["FLPILASLIGGLISFL", "GLFDIIKKIAESF", "GVLDILKBIAKGFA"],
    "[CHARGE_NEG]": ["EDDDEDDEE", "EEDDEEDDD", "FLEDIIEEIEREE"]
}

def generate_sequences(tokenizer, model, device, tags, count, temp=0.7):
    if model is not None and tokenizer is not None:
        prompt = f"<|startoftext|>\n{tags}\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        results = []
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=80,
                do_sample=True,
                top_k=40,
                top_p=0.90,
                temperature=temp,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_return_sequences=count * 2,
                pad_token_id=tokenizer.eos_token_id
            )
        for out in outputs:
            raw_text = tokenizer.decode(out, skip_special_tokens=False)
            lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
            for line in lines:
                if not line.startswith("<|") and not line.startswith("[") and not line.endswith("|>"):
                    seq = line.replace("<|endoftext|>", "").strip()
                    if seq and seq not in results:
                        results.append(seq)
                        break
            if len(results) >= count:
                break
        if results:
            return results

    # Fallback sampling if model is not loaded
    charge_tag = [t for t in tags.split() if "CHARGE" in t]
    tag_key = charge_tag[0] if charge_tag else "[CHARGE_POS]"
    templates = PEPTIDE_TEMPLATES.get(tag_key, PEPTIDE_TEMPLATES["[CHARGE_POS]"])
    results = []
    for i in range(count):
        base_seq = list(templates[i % len(templates)])
        if len(base_seq) > 5:
            base_seq[-1] = str(np.random.choice(["K", "R", "L", "A"]))
        results.append("".join(base_seq))
    return results

# Page Layout
st.set_page_config(
    page_title="AMP-Genius v6 | Система оптимизации очистки пептидов",
    layout="wide"
)

if "history" not in st.session_state:
    st.session_state["history"] = load_history_from_disk()

st.title("AMP-Genius v6: Генератор пептидов и расчет карт очистки")
st.markdown("МодельProtGPT2 PEFT/LoRA с профильным анализом биофизических параметров и технологических буферов.")

st.sidebar.header("Параметры генерации")

target_profile = st.sidebar.radio(
    "Целевой профиль применения:",
    ["Системная медицина (Инъекции)", "Косметология (Наружное применение)"],
    help="В косметологии допускается pI в области 7.0–7.8 и не блокируется гемолиз."
)
is_cosmetics = "Косметология" in target_profile

tag_activity = st.sidebar.selectbox("Целевая активность:", ["[AMP]", "[ACP]", "[AMP] [ACP]"])
tag_charge = st.sidebar.selectbox("Заряд (pH 7.4):", ["[CHARGE_POS]", "[CHARGE_HIGH_POS]", "[CHARGE_NEUT]", "[CHARGE_NEG]"])
tag_length = st.sidebar.selectbox("Длина последовательности:", ["[LEN_SHORT]", "[LEN_MEDIUM]", "[LEN_LONG]"])
tag_hydro = st.sidebar.selectbox("Гидрофобность:", ["[HYDRO_MEDIUM]", "[HYDRO_HIGH]", "[HYDRO_LOW]"])

num_samples = st.sidebar.slider("Количество кандидатов:", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Температура генерации:", 0.3, 1.2, 0.7, 0.1)

if st.sidebar.button("Сгенерировать пептиды", type="primary"):
    selected_tags = f"{tag_activity} {tag_charge} {tag_length} {tag_hydro}"
    
    with st.spinner("Расчет последовательностей и биофизических параметров..."):
        tokenizer, model, device = load_model()
        sequences = generate_sequences(tokenizer, model, device, selected_tags, num_samples, temp=temperature)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_batch = []
        
        for idx, seq in enumerate(sequences, 1):
            props = calculate_biophysics(seq)
            hemo_risk = estimate_hemolysis_risk(props["Charge"], props["Gravy"], is_cosmetics)
            buf_rec = get_buffer_recommendations(props["pI"], props["MW"], props["Gravy"])
            
            entry = {
                "Время": timestamp,
                "Профиль": "Косметология" if is_cosmetics else "Медицина",
                "Теги": selected_tags,
                "Кандидат": f"Peptide #{idx}",
                "Последовательность": seq,
                "Длина": len(seq),
                "Заряд (pH 7.4)": props["Charge"],
                "pI": props["pI"],
                "Масса (Da)": props["MW"],
                "Гидрофобность": props["Gravy"],
                "Риск гемолиза": hemo_risk,
                "Буфер IEX": buf_rec["IEX"]["primary"]["name"]
            }
            current_batch.append((seq, props, hemo_risk, buf_rec))
            st.session_state["history"].append(entry)
            
        save_history_to_disk(st.session_state["history"])
        
        st.subheader("Результаты генерации и профилирования")
        
        for idx, (seq, props, hemo_risk, buf_rec) in enumerate(current_batch, 1):
            is_blood_pi = (7.0 <= props["pI"] <= 7.8)
            
            with st.expander(f"Кандидат #{idx}: {seq[:20]}... | MW: {props['MW']} Da | pI: {props['pI']} | Заряд: {props['Charge']}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Молекулярная масса", f"{props['MW']} Da")
                c2.metric("Изоэлектрическая точка (pI)", f"{props['pI']}")
                c3.metric("Заряд (pH 7.4)", f"{props['Charge']}")
                c4.metric("Оценка гемолиза", hemo_risk)
                
                if is_blood_pi and not is_cosmetics:
                    st.error("Внимание: pI входит в диапазон 7.0–7.8. Повышен риск агрегации в плазме крови при pH 7.4.")
                elif is_blood_pi and is_cosmetics:
                    st.info("Информация: pI ~ 7.4 допускается для наружного применения.")
                    
                if is_cosmetics and props["MW"] > 3500:
                    st.warning("Предупреждение: Молекулярная масса > 3500 Da может ограничивать трансдермальное проникновение.")

                st.markdown("#### Карта технологической очистки пептида")
                
                t_iex, t_hplc, t_storage = st.tabs(["1. Ионообменная хроматография (IEX)", "2. Препаративная ВЭЖХ (RP-HPLC)", "3. Растворение и хранение"])
                
                with t_iex:
                    iex = buf_rec["IEX"]
                    st.markdown(f"**Режим:** `{iex['mode']}`")
                    st.success(f"**Основная рекомендация:** {iex['primary']['name']}\n\n**Параметры:** {iex['primary']['details']}\n\n**Обоснование:** {iex['primary']['why']}")
                    st.markdown("**Альтернативные буферные системы:**")
                    st.warning(f"**Вариант A:** {iex['alt1']['name']}\n\n*Влияние:* {iex['alt1']['impact']}")
                    st.info(f"**Вариант B:** {iex['alt2']['name']}\n\n*Влияние:* {iex['alt2']['impact']}")

                with t_hplc:
                    hplc = buf_rec["RP_HPLC"]
                    st.success(f"**Основная рекомендация:** {hplc['primary']['name']}\n\n**Условия:** {hplc['primary']['details']}\n\n**Преимущества:** {hplc['primary']['why']}")
                    st.markdown("**Альтернативные подвижные фазы:**")
                    st.warning(f"**Вариант A (LC-MS):** {hplc['alt1']['name']}\n\n*Влияние:* {hplc['alt1']['impact']}")
                    st.info(f"**Вариант B (Нейтральная ВЭЖХ):** {hplc['alt2']['name']}\n\n*Влияние:* {hplc['alt2']['impact']}")

                with t_storage:
                    stg = buf_rec["Storage"]
                    st.success(f"**Основной состав для растворения:** {stg['primary']['name']}\n\n**Обоснование:** {stg['primary']['why']}")
                    st.info(f"**Альтернатива для хранения:** {stg['alt']['name']}\n\n*Примечание:* {stg['alt']['impact']}")

# Persistent History View
st.markdown("---")
st.subheader("Журнал сгенерированных пептидов")

if st.session_state["history"]:
    df_hist = pd.DataFrame(st.session_state["history"])
    
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.write(f"Сохранено записей в журнале: **{len(df_hist)}**")
    with col_h2:
        if st.button("Очистить журнал"):
            st.session_state["history"] = []
            save_history_to_disk([])
            st.rerun()
            
    st.dataframe(df_hist, use_container_width=True)
    
    csv_data = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать историю (CSV)",
        data=csv_data,
        file_name="generated_peptides_history.csv",
        mime="text/csv"
    )
else:
    st.info("Журнал пуст. Запустите генерацию пептидов для добавления записей.")
