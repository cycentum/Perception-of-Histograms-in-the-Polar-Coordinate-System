<?php
function makePath($sessionID, $eqtid, $dataLength, $expType) {
	$path = "Result_{$expType}/{$eqtid}/{$sessionID}/{$dataLength}.json"; 
	return $path;
}


function numFiles($dirArray, $returnFileList)
{
	$result = array();

	foreach (array_keys($dirArray) as $key) {
		$dir=$dirArray[$key];

		if (is_dir($dir)) {
			$files = scandir($dir);
			$files = array_diff($files, array('.', '..')); // Remove . and ..

			if ($returnFileList) {
				$result[$key] = array_values($files);
			} else {
				$result[$key] = count($files);
			}
		}
		else {
			if ($returnFileList) {
				$result[$key] = [];
			} else {
				$result[$key] = 0;
			}
		}
	}

	return $result;
}


function minValue_randIfTie($array){
	$minValue = min($array);
	$minKeys = array_keys($array, $minValue);
	echo "console.log(`";var_dump($minKeys);echo "`);";
	if(count($minKeys)==1)
	{
		$minKey=$minKeys[0];
	}
	else
	{
		$minKey = $minKeys[mt_rand(0, count($minKeys)-1)];
	}
	return $minKey;
}

?>
