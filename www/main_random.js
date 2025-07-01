
function main_random()
{
	/* initialize jsPsych */
	jsPsych = initJsPsych();

	// const beforeunload_func=null;  //for debug
	const beforeunload_func=event=>event.preventDefault();
	window.addEventListener('beforeunload', beforeunload_func);

	history.pushState(null, null, location.href);
	window.addEventListener('popstate', event=>{
		history.go(1);
	});

	/* create timeline */
	let timeline = [];
	
	const params = new URLSearchParams(document.location.search);
	
	// const posFixed_all=["left", "right"];
	// const posFixed=parseParam(params, "posFixed", posFixed_all);
	
	// const eqType_all=["Len", "Area"];
	// const eqType=parseParam(params, "eqType", eqType_all);

	// const binSize_all=[6, 10, 14, 18];
	// const binSize=Utils.choice(binSize_all);

	// const peak_to_trough_all=["1.5", "2.0", "3.0", "5.0"];
	// const peak_to_trough=Utils.choice(peak_to_trough_all);

	// const setIndex=jsPsych.randomization.randomInt(0, numSet-1);

	// const expType="random";
	sessionParam=new SessionParam(expType, posFixed, eqType);
	sessionParam.binSize=binSize;
	sessionParam.peak_to_trough=peak_to_trough;
	sessionParam.setIndex=setIndex;
	console.table(sessionParam);
	
	trialInfo_now=TrialInfo.makeEmpty();
	
	let trial_init=new Trial_Init(params);
	timeline.push(trial_init);
	
	timeline.push(new Trial_BrowserCheck());

	const svgScale=1.0;

	{
		const blockType="training";
		timeline.push(new Trial_Title(blockType));
		
		const split_all=jsPsych.randomization.repeat(["0.2", "5.0"], 1);

		for(let ic=0; ic<coordType_all.length; ++ic)
			for(let f_b=0; f_b<2; ++f_b)
		{
			const coordType=coordType_all[ic];
			const split=split_all[f_b];
			const scale=1.0;

			let state_all;
			// if(f_b==0)
			{
				state_all=["Orig", "Shape", "Fixed", "Adjustable",];
			}
			// else if(f_b==1)
			// {
			//   state_all=["FixedAdjustable",];
			// }

			for(let st=0; st<state_all.length; ++st)
			{
				const state=state_all[st];
				let state_dirName;
				if(state=="Orig" || state=="Shape" )
				{
					state_dirName=state;
				}
				else if(state=="Fixed")
				{
					state_dirName="TextL_Flip"+{"left":"F", "right":"T"}[posFixed];
				}
				else if(state=="Adjustable")
				{
					state_dirName="TextR_Flip"+{"left":"F", "right":"T"}[posFixed];
				}
				// else if(state=="FixedAdjustable")
				// {
				//   state_dirName="TextLR";
				// }
				
				const path=`Stim_random/Bin${binSize}_PtF${peak_to_trough}/${eqType}_${posFixed}/Set${setIndex}/tutorial/${state_dirName}/Split${split}_${coordType}${eqType}_Ins0.svg`;
				const stimInfo=new TrialInfo(blockType, coordType, path, parseFloat(split), scale);
				stimInfo.state=state;

				timeline.push(new Trial_Preload(stimInfo));
				timeline.push(new Trial_Tutorial(stimInfo, state));
			}
		}
	}
	{
		const blockType="main";
		timeline.push(new Trial_Title(blockType));

		const split_all=["0.25", "0.5", "2.0", "4.0"];
		const numInstance=4;
		const instance_all=[...Array(numInstance).keys()];

		const factors={
			coordType: coordType_all,
			split:split_all, 
			instance:instance_all,};
		const stimParam_all=factorial_shuffle_noConsecutive(factors);
		console.table(stimParam_all);

		for(let sp=0; sp<stimParam_all.length; ++sp)
		{
			const stimParam=stimParam_all[sp];
			const split=stimParam.split;
			const coordType=stimParam.coordType;
			const instance=stimParam.instance;

			const path=`Stim_random/Bin${binSize}_PtF${peak_to_trough}/${eqType}_${posFixed}/Set${setIndex}/main/Split${split}_${coordType}${eqType}_Ins${instance}.svg`;
			const stimInfo=new TrialInfo(blockType, coordType, path, parseFloat(split), svgScale);
		
			timeline.push(new Trial_Preload(stimInfo));
			timeline.push(new Trial_Stim(stimInfo));
			timeline.push(new Trial_Save(sp==stimParam_all.length-1));
		}
	}

	console.log(isInternal(params), params.get("id"), params.get("eqtid"));
	timeline.push(new Trial_End(params, beforeunload_func, trial_init));

	/* start the experiment */
	jsPsych.run(timeline);
}
