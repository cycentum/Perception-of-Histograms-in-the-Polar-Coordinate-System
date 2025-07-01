<?php
include 'utils.php';

// get the data from the POST message
$data = json_decode(file_get_contents('php://input'), true);

// generate a unique ID for the file, e.g., session-6feu833950202 
$sessionID=$data[0]["sessionID"];
$expType=$data[0]["sessionParam"]["expType"];
$id=$data[0]["params"]["id"];
$eqtid=$data[0]["params"]["eqtid"];
$path=makePath($sessionID, $eqtid, count($data), $expType);
$dir=dirname($path);
if(!is_dir($dir))
{
	mkdir($dir, 0705, true);
}
// write the file to disk
$jsonData=json_encode($data, JSON_PRETTY_PRINT);
file_put_contents($path, $jsonData);

echo "Data saved to $path";

$completedState=$data[0]["completedState"];
if($completedState["completed"])
{
	foreach($completedState["path_all"] as $path)
	{
		$dir=dirname($path);
		if(!is_dir($dir))
		{
			mkdir($dir, 0705, true);
		}
		file_put_contents($path, "");

		echo "Made empty file: $path";
	}
}

?>