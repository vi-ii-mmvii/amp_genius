import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import datetime
import os

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

# Page Configuration
st.set_page_config(
    page_title="AMP-Genius | De Novo Peptide Generator",
    page_icon="🧬",
    layout="wide"
)

HYDROPHOBIC_RESIDUES = set("AILMFWV")

def calculate_net_charge(seq: str) -> float:
    seq = seq.upper()
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return round(pos - neg, 1)

def calculate_hydrophobicity_ratio(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    hydro_count = sum(1 for aa in seq if aa in HYDROPHOBIC_RESIDUES)
    return round(hydro_count / len(seq), 2)

def calculate_biophysical_properties(seq: str):
    seq_clean = "".join([aa for aa in seq.upper() if aa in "ACDEFGHIKLMNPQRSTVWY"])
    if not seq_clean:
        return 0.0, 0.0
    try:
        analyser = ProteinAnalysis(seq_clean)
        mw = round(analyser.molecular_weight(), 1)
        pI = round(analyser.isoelectric_point(), 2)
        return mw, pI
    except Exception:
        return 0.0, 0.0

@st.cache_resource
def load_model():
    base_model_name = "nferruz/ProtGPT2"
    adapter_path = "./amp_genius_lora_final"
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

# Initialize Session State
if "history" not in st.session_state:
    st.session_state["history"] = load_history_from_disk()

st.title("🧬 AMP-Genius: Generative Design of Dual-Action Peptides")
st.markdown("Генератор де-ново антимикробных и противоопухолевых пептидов на базе **ProtGPT2 + LoRA** с модулем **Rejection Sampling (Specific Filter)**")

# Sidebar - Condition Selection
st.sidebar.header("🎯 Выбор управляющих условий (Prompt Tags)")

activities = st.sidebar.multiselect(
    "Биологическая активность",
    options=["[AMP]", "[ACP]"],
    default=["[AMP]", "[ACP]"],
    help="[AMP] - Антимикробный пептид, [ACP] - Противоопухолевый пептид"
)

charge_option = st.sidebar.selectbox(
    "Заряд (pH 7.4)",
    options=["[CHARGE_HIGH_POS]", "[CHARGE_POS]", "[CHARGE_NEUT]", "[CHARGE_NEG]"],
    index=0
)

length_option = st.sidebar.selectbox(
    "Длина пептида",
    options=["[LEN_SHORT]", "[LEN_MEDIUM]", "[LEN_LONG]"],
    index=0,
    help="SHORT: <20 aa, MEDIUM: 20-35 aa, LONG: >35 aa"
)

hydro_option = st.sidebar.selectbox(
    "Гидрофобность",
    options=["[HYDRO_HIGH]", "[HYDRO_MEDIUM]", "[HYDRO_LOW]"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Настройки фильтрации и генерации")

strict_filter = st.sidebar.checkbox(
    "🛡️ Включить Specific Filter (Rejection Sampling)",
    value=True,
    help="Автоматически отбраковывает пептиды, которые не соответствуют выбору заряда или pI"
)

num_sequences = st.sidebar.slider("Количество кандидатов", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Температура генерации (Креативность)", 0.3, 1.2, 0.7, 0.1)

use_seed = st.sidebar.checkbox("Фиксировать seed (для воспроизводимости)", value=False)
seed_val = st.sidebar.number_input("Значение seed", value=42, step=1) if use_seed else None

if st.sidebar.button("🚀 Сгенерировать пептиды", type="primary"):
    selected_tags = "".join(activities) + charge_option + length_option + hydro_option
    prompt = f"<|startoftext|>\n{selected_tags}\n"
    
    st.info(f"**Сформированный управляющий промпт:** `{selected_tags}`")
    
    if use_seed and seed_val is not None:
        torch.manual_seed(seed_val)
    
    with st.spinner("⏳ Загрузка модели ProtGPT2 и отбор валидных пептидов..."):
        try:
            tokenizer, model, device = load_model()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            current_results = []
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            max_attempts = 5
            attempt_count = 0
            
            while len(current_results) < num_sequences and attempt_count < max_attempts:
                attempt_count += 1
                needed = num_sequences - len(current_results)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=80,
                        do_sample=True,
                        top_k=40,
                        top_p=0.90,
                        temperature=temperature,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        num_return_sequences=max(needed * 2, 4),
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                for out in outputs:
                    raw_text = tokenizer.decode(out, skip_special_tokens=False)
                    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
                    
                    seq = ""
                    for line in lines:
                        if not line.startswith("<|") and not line.startswith("[") and not line.endswith("|>"):
                            seq = line.replace("<|endoftext|>", "").strip()
                            break
                    
                    if seq:
                        charge = calculate_net_charge(seq)
                        hydro = calculate_hydrophobicity_ratio(seq)
                        mw, pI = calculate_biophysical_properties(seq)
                        
                        # Проверка Specific Filter (Rejection Sampling)
                        is_valid = True
                        mismatch_reason = ""
                        
                        if strict_filter:
                            if "[CHARGE_HIGH_POS]" in charge_option or "[CHARGE_POS]" in charge_option:
                                if charge <= 0:
                                    is_valid = False
                                    mismatch_reason = f"Отрицательный или нулевой заряд ({charge})"
                            elif "[CHARGE_NEG]" in charge_option:
                                if charge >= 0:
                                    is_valid = False
                                    mismatch_reason = f"Положительный заряд ({charge})"
                            
                            # Фильтр на pI (отсеиваем пептиды, выпадающие в осадок при pH 7.4)
                            if 7.0 <= pI <= 7.8:
                                is_valid = False
                                mismatch_reason = f"Опасный pI ({pI}) — риск выпадения в осадок при pH 7.4"
                        
                        if is_valid or not strict_filter:
                            # Проверяем, чтобы не было дублей в результатах текущего запуска
                            if not any(res["Последовательность"] == seq for res in current_results):
                                entry = {
                                    "Время": timestamp,
                                    "Промпт": selected_tags,
                                    "Кандидат": f"Peptide #{len(current_results) + 1}",
                                    "Последовательность": seq,
                                    "Длина (аа)": len(seq),
                                    "Заряд (pH 7.4)": f"+{charge}" if charge > 0 else str(charge),
                                    "Гидрофобность": f"{int(hydro * 100)}%",
                                    "pI": pI,
                                    "Масса (Da)": mw,
                                    "Статус": "✅ Match" if is_valid else f"⚠️ {mismatch_reason}"
                                }
                                current_results.append(entry)
                                st.session_state["history"].append(entry)
                        
                        if len(current_results) == num_sequences:
                            break
            
            save_history_to_disk(st.session_state["history"])
            
            if current_results:
                st.success(f"✨ Отобрано валидных кандидатов: {len(current_results)} (из {attempt_count} попыток генерации)")
                for row in current_results:
                    with st.expander(f"🧬 {row['Кандидат']}: {row['Последовательность']}", expanded=True):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Длина", f"{row['Длина (аа)']} aa")
                        col2.metric("Заряд (pH 7.4)", row['Заряд (pH 7.4)'])
                        col3.metric("Гидрофобность", row['Гидрофобность'])
                        col4.metric("pI (Изоточки)", row['pI'], help=">9.0 - отлично для крови, 7.0-7.8 - отсеивать")
                        col5.metric("Масса", f"{row['Масса (Da)']} Da")
                        
                        if "⚠️" in row['Статус']:
                            st.warning(f"Причина несоответствия: {row['Статус']}")
                        else:
                            st.caption("✅ Прошёл контроль Specific Filter (Rejection Sampling)")
            else:
                st.warning("Не удалось сгенерировать пептид, полностью удовлетворяющий строгим критериям. Попробуйте отключить Specific Filter или выставить температуру 0.7.")
                
        except Exception as e:
            st.error(f"Ошибка при работе модели: {e}")
            st.info("Убедитесь, что папка `amp_genius_lora_final` находится в корне проекта.")

# --- History Section ---
st.write("---")
st.header("📜 Локальная история всех сгенерированных пептидов")

if st.session_state["history"]:
    df_hist = pd.DataFrame(st.session_state["history"])
    
    col_h1, col_h2 = st.columns([4, 1])
    with col_h1:
        st.write(f"Всего сохранено кандидатов (включая прошлые запуски): **{len(df_hist)}**")
    with col_h2:
        if st.button("🗑️ Очистить историю"):
            st.session_state["history"] = []
            save_history_to_disk([])
            st.rerun()
            
    st.dataframe(df_hist, use_container_width=True)
    
    csv_data = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать всю историю пептидов (CSV)",
        data=csv_data,
        file_name="generated_peptides_history.csv",
        mime="text/csv"
    )
else:
    st.info("История пока пуста. Нажмите кнопку «🚀 Сгенерировать пептиды» выше, чтобы начать заполнение журнала.")