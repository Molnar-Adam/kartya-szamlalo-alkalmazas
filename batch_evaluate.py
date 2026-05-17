import argparse
import pandas as pd
from pathlib import Path
from single_card_recognition import recognize_single_card


"""
Batch kiértékelést futtat az Excelben megadott képlistán,
és összesíti a rank/suit pontosságot.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", required=True, type=Path, help="Az Excel fájl elérési útja")
    parser.add_argument("--images-dir", default=Path("images"), type=Path, help="A képek mappája")
    parser.add_argument("--rank-templates", default=Path("templates/rank"), type=Path)
    parser.add_argument("--suit-templates", default=Path("templates/suit"), type=Path)
    args = parser.parse_args()

    if not args.excel.exists():
        print(f"Nem található az Excel fájl: {args.excel}")
        return

    try:
        df = pd.read_excel(str(args.excel), header=None) 
    except Exception as e:
        print(f"Hiba az Excel fájl beolvasásakor: {e}")
        return

    correct_both = 0
    correct_rank = 0
    correct_suit = 0
    total = 0

    print("Képek feldolgozása...\n")

    for index, row in df.iterrows():
        filename = str(row.iloc[0]).strip()
        
        if not filename.lower().endswith('.jpg'):
            filename += '.jpg'
            
        expected_suit = str(row.iloc[1]).strip()
        expected_rank = str(row.iloc[2]).strip()
        
            
        total += 1
        img_path = args.images_dir / filename
        
        if not img_path.exists():
            print(f"HIÁNYZÓ KÉP: {filename} nem található itt: {img_path}")
            continue

        try:
            pred_rank, pred_suit = recognize_single_card(
                image_path=img_path,
                rank_templates_dir=args.rank_templates,
                suit_templates_dir=args.suit_templates,
                debug=False
            )
        except Exception as e:
            print(f"HIBA FELDOLGOZÁSKOR -> {filename}: {e}")
            continue

        if pred_rank == expected_rank:
            correct_rank += 1
        if pred_suit == expected_suit:
            correct_suit += 1

        if pred_rank == expected_rank and pred_suit == expected_suit:
            correct_both += 1
            print(f"OK -> {filename}: {pred_rank}_{pred_suit}")
        else:
            rank_status = "OK" if pred_rank == expected_rank else f"HIBA (várt: {expected_rank}, kapott: {pred_rank})"
            suit_status = "OK" if pred_suit == expected_suit else f"HIBA (várt: {expected_suit}, kapott: {pred_suit})"
            print(f"TÉVESZTÉS -> {filename} | Rank: {rank_status} | Suit: {suit_status}")

    if total > 0:
        accuracy_both = (correct_both / total) * 100
        accuracy_rank = (correct_rank / total) * 100
        accuracy_suit = (correct_suit / total) * 100
        print(f"\n==========================================")
        print(f"ÖSSZESÍTÉS ({total} feldolgozott elem alapján):")
        print(f"Rank helyes  : {correct_rank} / {total} -> {accuracy_rank:.2f}%")
        print(f"Suit helyes   : {correct_suit} / {total} -> {accuracy_suit:.2f}%")
        print(f"Teljes egyezés: {correct_both} / {total} -> {accuracy_both:.2f}%")
        print(f"==========================================")
    else:
        print("\nNem találtunk feldolgozható elemet.")

if __name__ == "__main__":
    main()