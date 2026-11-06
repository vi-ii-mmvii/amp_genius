import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd

# Конфигурация страницы Streamlit
st.set_page_config(
    page_title="AMP-Genius | De Novo Peptide Generator",
#    page_icon="🧬",
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

st.title("AMP-Genius: Generative Design of Dual-Action Peptides")
st.markdown("Генератор де-ново антимикробных и противоопухолевых пептидов на базе **ProtGPT2 + LoRA**")

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

num_sequences = st.sidebar.slider("Количество кандидатов", min_value=1, max_value=5, value=3)
temperature = st.sidebar.slider("Температура генерации (Креативность)", 0.3, 1.2, 0.7, 0.1)

# Кнопка генерации
if st.sidebar.button("🚀 Сгенерировать пептиды", type="primary"):
    selected_tags = "".join(activities) + charge_option + length_option + hydro_option
    prompt = f"<|startoftext|>\n{selected_tags}\n"
    
    st.info(f"**Сформированный управляющий промпт:** `{selected_tags}`")
    
    with st.spinner("⏳ Загрузка модели ProtGPT2 и генерация последовательностей..."):
        try:
            tokenizer, model, device = load_model()
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=90,
                    do_sample=True,
                    top_k=50,
                    top_p=0.92,
                    temperature=temperature,
                    num_return_sequences=num_sequences,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            results = []
            for i, out in enumerate(outputs, 1):
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
                    results.append({
                        "Кандидат": f"Peptide #{i}",
                        "Последовательность": seq,
                        "Длина (аа)": len(seq),
                        "Заряд (pH 7.4)": f"+{charge}" if charge > 0 else str(charge),
                        "Гидрофобность": f"{int(hydro * 100)}%"
                    })
            
            if results:
                st.success("Генерация успешно завершена!")
                df_res = pd.DataFrame(results)
                
                for row in results:
                    with st.expander(f"genes: {row['Кандидат']}: {row['Последовательность']}", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Длина", f"{row['Длина (аа)']} aa")
                        col2.metric("Заряд", row['Заряд (pH 7.4)'])
                        col3.metric("Гидрофобность", row['Гидрофобность'])
                
                st.dataframe(df_res, use_container_width=True)
            else:
                st.warning("Модель сгенерировала маркер конца текста. Попробуйте увеличить температуру или изменить теги.")
                
        except Exception as e:
            st.error(f"Ошибка при работе модели: {e}")
            st.info("Убедитесь, что папка `amp_genius_lora_final` находится в корне проекта.")
