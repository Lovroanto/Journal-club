from pathlib import Path
import re

text = Path("/home/lbarisic/ai_data/Journal_Club/First/Newversion/Summary/slides/003_Introduction_to_Continuous_Lasing.txt").read_text()

print(re.search(r"===\s*SLIDE_GROUP:\s*(.*?)\s*\|", text))