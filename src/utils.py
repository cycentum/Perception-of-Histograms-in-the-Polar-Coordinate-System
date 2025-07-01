from io import BytesIO
import pickle
import json


def copyByte(obj):
	byt=BytesIO()
	pickle.dump(obj, byt)
	
	byt=BytesIO(byt.getvalue())
	obj_copy=pickle.load(byt)
	
	return obj_copy


def load_or_dump_rng(fileRNG, rng):
	if fileRNG.is_file():
		with open(fileRNG, "rb") as f: rng=pickle.load(f)
	else:
		with open(fileRNG, "wb") as f: pickle.dump(rng, f)
	return rng


def write_json_as_jsVar(file, varName, data):
	with open(file, "w") as f:
		f.write(f"var {varName} = ")
		json.dump(data, f, indent=1)
		f.write(";")
		