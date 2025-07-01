<!DOCTYPE html>
<html>
	<head>
		<title></title>
		<meta charset="UTF-8">

		<script src="https://unpkg.com/jspsych@8.0.0"></script>
		<script src="https://unpkg.com/@jspsych/plugin-html-keyboard-response@2.0.0"></script>
		<script src="https://unpkg.com/@jspsych/plugin-preload@2.0.0"></script>
		<script src="https://unpkg.com/@jspsych/plugin-call-function@2.0.0"></script>
		<script src="https://unpkg.com/@jspsych/plugin-browser-check@2.0.0"></script>
		<link href="https://unpkg.com/jspsych@8.0.0/css/jspsych.css" rel="stylesheet" type="text/css" />
		
		<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/stdlib-js/random-sample@umd/browser.js"></script>
		<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/stdlib-js/random-shuffle@umd/browser.js"></script>
		<script type="text/javascript" src="https://cdn.jsdelivr.net/gh/stdlib-js/array-incrspace@umd/browser.js"></script>
		

<style>
body{
	-webkit-user-select: none;  /* Chrome all / Safari all */
	-moz-user-select: none;     /* Firefox all */
	-ms-user-select: none;      /* IE 10+ */
	 user-select: none;       
}

.p{
	font-weight: normal;
}

.b
{
	font-weight: bold;
	font-size: 110%;
}

.small
{
	font-size: 80%;
}

#stimId{
	display:block;
}


#container-flex
{
	display: flex;
	justify-content: center;
	align-items: center;
	width:87.5svw;
	height: 87.5svh;

	/* background-color: blue; */
}


#container-center
{
	width: 100%;

	/*background-color: green;*/
}


@media (max-aspect-ratio: 4/5) {
	.p {
		font-size: 2.3svw;
		line-height: 2.5svw;
	}
}

@media (min-aspect-ratio: 4/5) {
	.p {
		font-size: calc((2.3svh)*(4/5));
		line-height: calc((2.5svh)*(4/5));
	}
}


.shake {
animation: shake 0.25s;
}


@keyframes shake {
	10% { transform: translate(0.2em, -0.4em) rotate(1deg); }
	20% { transform: translate(-0.4em, 0.2em) rotate(0deg); }
	30% { transform: translate(0.4em, 0) rotate(-1deg); }
	40% { transform: translate(-0.2em, 0.2em) rotate(0deg); }
	50% { transform: translate(0, -0.4em) rotate(1deg); }
	60% { transform: translate(-0.2em, 0.4em) rotate(0deg); }
	70% { transform: translate(0.4em, -0.2em) rotate(-1deg); }
	80% { transform: translate(-0.4em, 0) rotate(0deg); }
	90% { transform: translate(0.2em, -0.2em) rotate(1deg); }
	100% { transform: translate(0, 0.4em) rotate(0deg); }
}


#overlay_forceFullscreen{
	position:fixed;
	top:0;
	left:0;
	width:100%;
	height:100%;
	background-color:#ffffff;
	/* z-index:0; */
	display: none;
}

#forceFullscreen-content{
	position: absolute;
	top: 50%;
	left: 50%;
	transform: translate(-50%, -50%);
	text-align: center;
	font-size:xx-large;
}


#stim-container-outmost{
	line-height:0;
	position:relative;
	margin: 0 auto;

	/* background-color: red; */
}

#stimSVG{
	position:absolute;
	top:0;
	left:0;
	width:100%;
	height:100%;

	/* border: 1px solid red; */
}

.td{
	display: inline-block;
	/* width:12em; width is adjusted in common.js -> set_svgPos()*/
	text-align: center;
}


.hidden{
	visibility: hidden;
}

.lonlyButton{
	padding:2em;
}


#afterFinish{
	display:none;
}

</style>

	<script type="text/javascript">

		<?php 
			require "./utils.php";

			$r = mt_rand(0, 8); //inclusive; sizes=9
			if($r==0) // prob = 1/9
			{
				$expType="singlePeak";
			}
			else
			{
				$expType="random";
			}

			$eqtid=$_GET["eqtid"];

			$eqType_all=["Len", "Area"];
			$posFixed_all=["left", "right"];

			if($expType=="singlePeak")
			{
				$completedDir=[];
				foreach ($eqType_all as $eqType) foreach ($posFixed_all as $posFixed) {
					$completedDir["{$eqType}_{$posFixed}"] = "./Completed_{$expType}/{$eqtid}/eqType_posFixed/{$eqType}_{$posFixed}";
				}
				$completedSize=numFiles($completedDir, false); #{eqType: size}
				echo "console.log(`";var_dump($completedSize);echo "`);";
				$eqType_posFixed=minValue_randIfTie($completedSize);
				list($eqType, $posFixed) = explode('_', $eqType_posFixed);
			}
			else if($expType=="random")
			{
				$binSize_all=["6", "10", "14", "18"];
				$peak_to_trough_all=["1.5", "2.0", "3.0", "5.0"];
				$completedDir=[];
				foreach($binSize_all as $binSize)  foreach($peak_to_trough_all as $peak_to_trough)
				{
					$completedDir["{$binSize}_{$peak_to_trough}"]="./Completed_{$expType}/{$eqtid}/Bin_PtF/Bin{$binSize}_Ptf{$peak_to_trough}";
				}
				$completedSize=numFiles($completedDir, false);
				echo "console.log(`";var_dump($completedSize);echo "`);";
				$binSize_PtF=minValue_randIfTie($completedSize);
				list($binSize, $peak_to_trough) = explode('_', $binSize_PtF);

				$completedDir=[];
				foreach ($eqType_all as $eqType) foreach ($posFixed_all as $posFixed) {
					$completedDir["{$eqType}_{$posFixed}"] = "./Completed_{$expType}/{$eqtid}/eqType_posFixed/Bin{$binSize}_Ptf{$peak_to_trough}/{$eqType}_{$posFixed}";
				}
				$completedSize=numFiles($completedDir, false); #{eqType: size}
				echo "console.log(`";var_dump($completedSize);echo "`);";
				$eqType_posFixed=minValue_randIfTie($completedSize);
				list($eqType, $posFixed) = explode('_', $eqType_posFixed);

				require "./Stim_random/numSet.php";
				$completedDir=["./Completed_{$expType}/{$eqtid}/SetIndex/Bin{$binSize}_Ptf{$peak_to_trough}/{$eqType}_{$posFixed}", ];
				$completedFiles = numFiles($completedDir, true)[0];
				echo "console.log(`";var_dump($completedFiles);echo "`);";

				$completedFilesArray = array_fill_keys($completedFiles, true);
				$setIndex_01 = array_map(function($index) use ($completedFilesArray) {
					return (int)array_key_exists("Set{$index}", $completedFilesArray);
				}, range(0, $numSet - 1)); // 0 if not exist, 1 if exist in $completedFilesArray
				$setIndex = minValue_randIfTie($setIndex_01);
			}

			echo <<<END
			var expType="{$expType}";
			console.log({expType});
			var eqType="{$eqType}";
			console.log({eqType});
			var posFixed="{$posFixed}";
			console.log({posFixed});
			END;

			if($expType=="random")
			{
				echo <<<END
				var binSize="{$binSize}";
				console.log({binSize});
				var peak_to_trough="{$peak_to_trough}";
				console.log({peak_to_trough});
				var setIndex="{$setIndex}";
				console.log({setIndex});
				END;
			}
		?>

	</script>

	<script type="text/javascript" src="./canvSize.js"></script>
	
	<script type="text/javascript" src="./utils.js"></script>

	<script type="text/javascript" src="./common.js"></script>
	<script type="text/javascript" src="./main_<?php echo $expType;?>.js"></script>

	<script type="text/javascript">
		document.addEventListener("DOMContentLoaded", main_<?php echo $expType;?>);
	</script>

	</head>
	<body>

<div id="overlay_forceFullscreen">
		<div id="forceFullscreen-content" class="p">
			Enterキーを押して進む<br><br>全画面表示になります
		</div>
</div>

	</body>
</html>
