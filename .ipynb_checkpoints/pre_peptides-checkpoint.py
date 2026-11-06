#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = set("AILMFWV")

def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 50) -> bool:
    """Проверка на канонические аминокислоты и допустимую длину."""
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    return (min_len <= len(seq) <= max_len) and set(seq).issubset(STANDARD_AMINO_ACIDS)

def calculate_net_charge(seq: str) -> float:
    """Расчёт чистого заряда при pH 7.4."""
    seq = seq.upper()
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return round(pos - neg, 1)

def calculate_hydrophobicity_ratio(seq: str) -> float:
    """Расчёт доли гидрофобных аминокислот."""
    seq = seq.upper()
    hydro_count = sum(1 for aa in seq if aa in HYDROPHOBIC_RESIDUES)
    return round(hydro_count / len(seq), 2)

def derive_condition_tags(seq: str, is_amp: bool = False, is_acp: bool = False) -> List[str]:
    """Генерация дискретных управляющих тегов."""
    tags = []
    if is_amp: tags.append("[AMP]")
    if is_acp: tags.append("[ACP]")
        
    charge = calculate_net_charge(seq)
    if charge >= 4.0: tags.append("[CHARGE_HIGH_POS]")
    elif charge > 0: tags.append("[CHARGE_POS]")
    elif charge == 0: tags.append("[CHARGE_NEUT]")
    else: tags.append("[CHARGE_NEG]")
        
    length = len(seq)
    if length < 20: tags.append("[LEN_SHORT]")
    elif length <= 35: tags.append("[LEN_MEDIUM]")
    else: tags.append("[LEN_LONG]")
    
    hydro_ratio = calculate_hydrophobicity_ratio(seq)
    if hydro_ratio >= 0.5: tags.append("[HYDRO_HIGH]")
    elif hydro_ratio >= 0.3: tags.append("[HYDRO_MEDIUM]")
    else: tags.append("[HYDRO_LOW]")
    
    return tags

def format_protgpt2_prompt(seq: str, tags: List[str]) -> str:
    """Форматирование промпта для ProtGPT2."""
    tag_prefix = "".join(tags)
    return f"<|startoftext|>\n{tag_prefix}\n{seq}\n<|endoftext|>"

def load_sequences(filepath: str) -> List[str]:
    """Чтение файлов CAMP3 и CancerPPD с корректной обработкой табуляции."""
    if not filepath or not os.path.exists(filepath):
        print(f"⚠️ Файл не найден: {filepath}")
        return []
    
    try:
        # Чтение файла с табуляцией (\t)
        df = pd.read_csv(filepath, sep='\t', on_bad_lines='skip')
    except Exception:
        try:
            df = pd.read_csv(filepath, sep=r'\s+', engine='python', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(filepath, on_bad_lines='skip')
    
    seq_col = None
    for col in df.columns:
        clean_col = str(col).strip().lower()
        if clean_col in ['seqence', 'sequence', 'seq', 'peptide', 'peptide_sequence']:
            seq_col = col
            break
            
    if seq_col is None:
        seq_col = df.columns[-1]
        
    sequences = []
    for raw_seq in df[seq_col].dropna():
        s = str(raw_seq).strip().upper()
        if is_valid_sequence(s):
            sequences.append(s)
            
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Preprocess peptides for ProtGPT2 fine-tuning")
    parser.add_argument("--camp3", type=str, required=True, help="Path to CAMP3 dataset")
    parser.add_argument("--cancerppd", type=str, required=True, help="Path to CancerPPD dataset")
    parser.add_argument("--outdir", type=str, default="data_processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("📥 Загрузка и очистка файлов...")
    camp_seqs = set(load_sequences(args.camp3))
    cancer_seqs = set(load_sequences(args.cancerppd))

    print(f"  • Найдено валидных пептидов в CAMP3: {len(camp_seqs)}")
    print(f"  • Найдено валидных пептидов в CancerPPD: {len(cancer_seqs)}")

    all_seqs = camp_seqs.union(cancer_seqs)
    
    records = []
    prompts = []

    for seq in all_seqs:
        is_amp = seq in camp_seqs
        is_acp = seq in cancer_seqs
        tags = derive_condition_tags(seq, is_amp=is_amp, is_acp=is_acp)
        prompt = format_protgpt2_prompt(seq, tags)
        
        records.append({
            "sequence": seq,
            "is_amp": is_amp,
            "is_acp": is_acp,
            "charge": calculate_net_charge(seq),
            "length": len(seq),
            "tags": "".join(tags),
            "prompt": prompt
        })
        prompts.append({"prompt": prompt})

    df_full = pd.DataFrame(records)
    train_prompts, val_prompts = train_test_split(prompts, test_size=0.1, random_state=42)

    pd.DataFrame(train_prompts).to_json(
        os.path.join(args.outdir, "train_peptides.jsonl"), orient="records", lines=True, force_ascii=False
    )
    pd.DataFrame(val_prompts).to_json(
        os.path.join(args.outdir, "val_peptides.jsonl"), orient="records", lines=True, force_ascii=False
    )
    df_full.to_csv(os.path.join(args.outdir, "full_processed_peptides.csv"), index=False)

    print(f"\n🎉 Успешно обработано!")
    print(f"📊 Всего уникальных пептидов: {len(all_seqs)}")
    print(f"💾 Файлы созданы в папке '{args.outdir}/':")
    print(f"   ├── train_peptides.jsonl ({len(train_prompts)} строк)")
    print(f"   ├── val_peptides.jsonl ({len(val_prompts)} строк)")
    print(f"   └── full_processed_peptides.csv")

if __name__ == "__main__":
    main()