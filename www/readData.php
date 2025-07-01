<?php
include 'utils.php';

// Path to the JSON file
$sessionID = $_GET['sessionID'] ?? null;
$eqtid = $_GET['eqtid'] ?? null;
$dataLength = $_GET['dataLength'] ?? null;
$expType = $_GET['expType'] ?? null;
$path=makePath($sessionID, $eqtid, $dataLength, $expType);

// Check if the file exists
if (file_exists($path)) {
	// Read the file contents
	$jsonData = file_get_contents($path);

	// Set the content type to application/json
	header('Content-Type: application/json');

	// Output the JSON data
	echo $jsonData;
} else {
	// If the file does not exist, send a 404 response
	http_response_code(404);
	echo json_encode(['error' => "File not found {$path}"]);
}
?>