import os
import lzma
import tempfile
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from nistrng import (
    check_eligibility_all_battery,
    run_all_battery,
    SP800_22R1A_BATTERY
)

_SBOX = np.array(
    np.random.default_rng(0xDEAD_BEEF).permutation(256),
    dtype=np.uint8
)

class BlindTestExperiment:

    @staticmethod
    def generate_noise(size_bytes):
        return bytearray(os.urandom(size_bytes))

    @staticmethod
    def generate_weak_sequence(size_bytes):
        data = bytearray(os.urandom(size_bytes))

        tap1 = 1
        tap2 = 2
        mask = np.uint8(0x03)

        for i in range(tap2, size_bytes):
            ctx = (data[i - tap1] ^ data[i - tap2]) & 0xFF
            data[i] = (
                (data[i] & 0xFC) |
                (_SBOX[ctx] & mask)
            )

        return data

    @staticmethod
    def get_lzma_size(data):
        return len(lzma.compress(data))

    @staticmethod
    def get_zpaq_size(data):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            tmp = f.name

        archive = tmp + ".zpaq"

        try:
            subprocess.run(
                ["zpaq", "add", archive, tmp, "-m5"],
                capture_output=True
            )
            return os.path.getsize(archive)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(archive):
                os.remove(archive)


    @staticmethod
    def prepare_bit_sequence(data, lsb_only=False):
        if not lsb_only:
            chunk = data[:125_000]
            seq = np.unpackbits(
                np.frombuffer(chunk, dtype=np.uint8)
            ).astype(int)
            return seq

        chunk = data[:500_000]
        arr = np.frombuffer(chunk, dtype=np.uint8)

        lsb1 = arr & 1
        lsb2 = (arr >> 1) & 1

        seq = np.empty(arr.size * 2, dtype=int)
        seq[0::2] = lsb2
        seq[1::2] = lsb1

        return seq

    @staticmethod
    def run_nist(data, lsb_only=False):
        seq = BlindTestExperiment.prepare_bit_sequence(
            data,
            lsb_only=lsb_only
        )

        eligible = check_eligibility_all_battery(
            seq,
            SP800_22R1A_BATTERY
        )

        results = run_all_battery(
            seq,
            eligible,
            False
        )
        return results

    @staticmethod
    def calculate_pass_rate(results):
        passed = sum(
            1 for result, _ in results if result.passed
        )
        total = len(results)
        return passed / total if total else 0.0

    @staticmethod
    def print_nist_details(results, title):
        print(f"\n{title}")
        print("=" * 95)
        print(
            f"{'Тест':<45} | "
            f"{'Статус':<12} | "
            f"{'p-value (score)'}"
        )
        print("-" * 95)

        for result, elapsed_time in results:
            status = (
                "ПРОЙДЕНО"
                if result.passed
                else "ПРОВАЛЕНО"
            )

            score = np.array(result.score)
            if score.size == 1:
                p_value_str = f"{score.item():.6f}"
            else:
                p_value_str = f"[Масив з {score.size} значень, сер. {score.mean():.4f}]"

            print(
                f"{result.name:<45} | "
                f"{status:<12} | "
                f"{p_value_str}"
            )

    @staticmethod
    def build_test_dataframe(results):
        rows = []
        for result, elapsed_time in results:
            score = np.array(result.score)
            val = score.item() if score.size == 1 else score.mean()
            
            rows.append({
                "Test": result.name,
                "Passed": result.passed,
                "p_value_mean": val
            })
        return pd.DataFrame(rows)


def plot_results(df):
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.boxplot(
            [
                df['NIST_Noise'],
                df['NIST_Weak'],
                df['NIST_Noise_LSB'],
                df['NIST_Weak_LSB']
            ],
            tick_labels=[
                'Еталон\n(8 біт)',
                'Модифікована\n(8 біт)',
                'Еталон\n(2 біти)',
                'Модифікована\n(2 біти)'
            ],
            patch_artist=True,
            boxprops=dict(facecolor='#90CAF9')
        )

    ax1.set_title(
        'NIST SP 800-22',
        fontsize=12
    )

    ax1.set_ylabel(
        'Частка успішно пройдених тестів'
    )

    ax1.grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    ax2.plot(
        df['Iteration'],
        df['CR_LZMA'],
        marker='o',
        linewidth=2,
        label='LZMA'
    )

    ax2.plot(
        df['Iteration'],
        df['CR_ZPAQ'],
        marker='s',
        linewidth=2,
        label='ZPAQ'
    )

    ax2.fill_between(
        df['Iteration'],
        df['CR_ZPAQ'],
        df['CR_LZMA'],
        alpha=0.2,
        label='Різниця'
    )

    ax2.set_title(
        'Нормалізовані коефіцієнти стиснення',
        fontsize=12
    )

    ax2.set_xlabel('Ітерація')
    ax2.set_ylabel('CR_norm')
    ax2.set_xticks(df['Iteration'])
    ax2.legend()

    ax2.grid(
        True,
        linestyle='--',
        alpha=0.6
    )

    plt.tight_layout()

    plt.savefig(
        'vulnerability_detection.png',
        dpi=300
    )

    plt.close()

    print(
        "\nГрафік збережено як "
        "'vulnerability_detection.png'"
    )

def main():

    FILE_SIZE = 2 * 1024 * 1024
    NUM_ITERATIONS = 30


    print("\n=== ДЕТАЛЬНИЙ NIST-АНАЛІЗ ===")

    sample = BlindTestExperiment.generate_weak_sequence(
        FILE_SIZE
    )

    full_results = BlindTestExperiment.run_nist(
        sample,
        lsb_only=False
    )

    BlindTestExperiment.print_nist_details(
        full_results,
        "Результати для повної послідовності (8 біт)"
    )

    lsb_results = BlindTestExperiment.run_nist(
        sample,
        lsb_only=True
    )

    BlindTestExperiment.print_nist_details(
        lsb_results,
        "Результати лише для 2 молодших бітів"
    )

    df_full = BlindTestExperiment.build_test_dataframe(
        full_results
    )

    df_lsb = BlindTestExperiment.build_test_dataframe(
        lsb_results
    )

    df_full.to_csv(
        "nist_full_sequence.csv",
        index=False
    )

    df_lsb.to_csv(
        "nist_lsb_only.csv",
        index=False
    )

    print("\nCSV-файли збережено:")
    print("- nist_full_sequence.csv")
    print("- nist_lsb_only.csv")

    results = []

    print(
        f"\nПочаток експерименту "
        f"({NUM_ITERATIONS} ітерацій)"
    )

    for i in range(NUM_ITERATIONS):

        print(
            f"Ітерація "
            f"{i + 1}/{NUM_ITERATIONS}"
        )

        noise = BlindTestExperiment.generate_noise(
            FILE_SIZE
        )

        weak = BlindTestExperiment.generate_weak_sequence(
            FILE_SIZE
        )

        nist_noise_results = BlindTestExperiment.run_nist(
            noise,
            lsb_only=False
        )

        nist_weak_results = BlindTestExperiment.run_nist(
            weak,
            lsb_only=False
        )

        nist_noise = (
            BlindTestExperiment.calculate_pass_rate(
                nist_noise_results
            )
        )

        nist_weak = (
            BlindTestExperiment.calculate_pass_rate(
                nist_weak_results
            )
        )

        nist_noise_lsb_results = BlindTestExperiment.run_nist(
            noise,
            lsb_only=True
        )

        nist_weak_lsb_results = BlindTestExperiment.run_nist(
            weak,
            lsb_only=True
        )

        nist_noise_lsb = (
            BlindTestExperiment.calculate_pass_rate(
                nist_noise_lsb_results
            )
        )

        nist_weak_lsb = (
            BlindTestExperiment.calculate_pass_rate(
                nist_weak_lsb_results
            )
        )

        ref_lzma = BlindTestExperiment.get_lzma_size(
            noise
        )

        ref_zpaq = BlindTestExperiment.get_zpaq_size(
            noise
        )

        weak_lzma = BlindTestExperiment.get_lzma_size(
            weak
        )

        weak_zpaq = BlindTestExperiment.get_zpaq_size(
            weak
        )

        cr_lzma = weak_lzma / ref_lzma
        cr_zpaq = weak_zpaq / ref_zpaq

        delta = cr_lzma - cr_zpaq

        results.append({
            'Iteration': i + 1,

            'NIST_Noise': nist_noise,
            'NIST_Weak': nist_weak,

            'NIST_Noise_LSB': nist_noise_lsb,
            'NIST_Weak_LSB': nist_weak_lsb,

            'Noise_ZPAQ_Size': ref_zpaq,
            'Weak_ZPAQ_Size': weak_zpaq,

            'Noise_LZMA_Size': ref_lzma,
            'Weak_LZMA_Size': weak_lzma,

            'CR_LZMA': cr_lzma,
            'CR_ZPAQ': cr_zpaq,
            'Delta': delta
        })

    df = pd.DataFrame(results)

    stat, p_value = wilcoxon(
        df['Noise_ZPAQ_Size'],
        df['Weak_ZPAQ_Size']
    )

    wins = np.sum(
        df['Weak_ZPAQ_Size']
        <
        df['Noise_ZPAQ_Size']
    )

    print("\n--- ПІДСУМКОВІ РЕЗУЛЬТАТИ ---")

    print(
        f"NIST (noise): "
        f"{df['NIST_Noise'].median():.3f}"
    )

    print(
        f"NIST (weak):  "
        f"{df['NIST_Weak'].median():.3f}"
    )

    print(
        f"NIST LSB (noise): "
        f"{df['NIST_Noise_LSB'].median():.3f}"
    )

    print(
        f"NIST LSB (weak):  "
        f"{df['NIST_Weak_LSB'].median():.3f}"
    )

    print(
        f"LZMA: "
        f"{df['CR_LZMA'].median():.6f}"
    )

    print(
        f"ZPAQ: "
        f"{df['CR_ZPAQ'].median():.6f}"
    )

    print(
        f"Δ = "
        f"{df['Delta'].median():.6f}"
    )

    print(
        f"ZPAQ виявив залежність у "
        f"{wins}/{NUM_ITERATIONS} запусків"
    )

    print(
        f"Wilcoxon p-value = "
        f"{p_value:.10f}"
    )

    plot_results(df)

if __name__ == "__main__":
    main()