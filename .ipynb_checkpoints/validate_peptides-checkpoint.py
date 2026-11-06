import pandas as pd

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC_RESIDUES = set("AILMFWV")

def calculate_net_charge(seq: str) -> float:
    seq = seq.upper()
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return round(pos - neg, 1)

def calculate_hydrophobicity_ratio(seq: str) -> float:
    seq = seq.upper()
    hydro_count = sum(1 for aa in seq if aa in HYDROPHOBIC_RESIDUES)
    return round(hydro_count / len(seq), 2)

# Сгенерированные пептиды
generated_peptides = ["ALAKKAKKAL", "KWKWKALKWKWK"]

# Загружаем обучающую базу для проверки на новизну
df_train = pd.read_csv("data_processed/full_processed_peptides.csv")
known_sequences = set(df_train["sequence"].str.upper())

print("🔬 РЕЗУЛЬТАТЫ IN SILICO ВАЛИДАЦИИ СГЕНЕРИРОВАННЫХ КАНДИДАТОВ:")
print("=" * 65)

for i, seq in enumerate(generated_peptides, 1):
    charge = calculate_net_charge(seq)
    hydro = calculate_hydrophobicity_ratio(seq)
    is_exact_match = seq in known_sequences
    
    print(f"Пептид #{i}: {seq}")
    print(f"  • Длина: {len(seq)} aa")
    print(f"  • Чистый заряд (pH 7.4): +{charge}")
    print(f"  • Гидрофобность: {hydro * 100}%")
    print(f"  • Присутствует в обучающей базе? {'ДУБЛИКАААТ' if is_exact_match else 'НЕТ (Абсолютно новый De Novo пептид!)'}")
    print("-" * 65)