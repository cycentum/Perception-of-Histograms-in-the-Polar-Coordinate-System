import subprocess
from pathlib import Path
import socket

analysisType_formulaName={
	"singlePeak":[
		"chosenRatio_FixEqCoord_RanSessionBinDist",
		"absError_FixEqCoord_RanSessionBinDistTrue",
	],
	"random":[
		"chosenRatio_FixEqCoord_RanSessionBinPtt",
		"absError_FixEqCoord_RanSessionBinPttTrue",
	],
}

dirProject=str(Path(__file__).parent.parent.parent)

# ratioType="true"
ratioType="expected"

for analysisType, formulaName_all in analysisType_formulaName.items():
	for formulaName in formulaName_all:
		subprocess.run(["Rscript", "LMEM.r", analysisType, formulaName, dirProject, ratioType], cwd=Path(__file__).parent)
		