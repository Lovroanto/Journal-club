from grobid_preprocessor import preprocess_pdf

pdf_path = "/home/lbarisic/ai_data/Journal_Club/First/summaries/references/articles/2012.08885v5.pdf"
output_dir = "/home/lbarisic/ai_data/Journal_Club/First/raw_test/"

results = preprocess_pdf(pdf_path, output_dir)

print("\n🎉 RAW GROBID OUTPUT")
print("====================")
print(f"TEI XML : {results['raw_tei']}")
print(f"TEXT    : {results['raw_text']}")
print("\nNow open the TEI XML to see what GROBID really extracted.")
