var jsPsych;
var svgPos_allScale=null; //svgPos[scale]
var sessionParam;
var trialInfo_now;
var chosenOptionHistory=null;
var timeHistory=null;
var currentDisplayType=null;
// const svgScale_all=[1, 1-1/16, ];
const svgScale_all=[1, ];
const coordType_all=["Cart", "Polar"];
var keyOn=false;
const delta=5;
var firstFullscreenRequest=true;
var timeout_keyRepeat=null;


class SessionParam{
  constructor(expType, posFixed, eqType){
    this.expType=expType;
    this.posFixed=posFixed;
    this.eqType=eqType;
  }
}


class TrialInfo{
  constructor(blockType, coordType, paths, splitRatio, svgScale){
    this.blockType=blockType;
    this.coordType=coordType;
    this.paths=paths;
    this.splitRatio=splitRatio;
    this.svgScale=svgScale;
  }

  copyFrom(orig)
  {
    this.blockType=orig.blockType;
    this.coordType=orig.coordType;
    this.paths=orig.paths;
    this.splitRatio=orig.splitRatio;
    this.svgScale=orig.svgScale;
  }

  static makeEmpty()
  {
    return new TrialInfo(null, null, null, null, null);
  }
}


function latestOption()
{
  if(chosenOptionHistory==null || chosenOptionHistory.length==0) return null;
  return chosenOptionHistory[chosenOptionHistory.length-1];
}


function initHistory()
{
  chosenOptionHistory=[];
  timeHistory=[];
  pushTimeHistory("Trial.on_load");
}


function saveHistory(data)
{
  if(chosenOptionHistory==null || chosenOptionHistory.length>0)  //Trial_Stim
  {
    data.chosenOptionHistory=chosenOptionHistory.slice();
  }
  data.timeHistory=timeHistory.slice();
}


function pushTimeHistory(type, obj)
{
  if(timeHistory==null) return;
  if(obj==null) obj={};
  obj.type=type
  obj.time=jsPsych.getTotalTime();
  timeHistory.push(obj);
}


function checkFullscreen_bySize()
{
  return screen.width==window.outerWidth && screen.height==window.outerHeight;
}


function run_requestFullscreen() { //run_ is needed because "requestFullscreen" is a builtin function name
  document.documentElement.requestFullscreen();
}


function isNotFullscreen()
{
  // return document.fullscreenElement == null || !checkFullscreen_bySize();
  return document.fullscreenElement == null;
}


function currentDisplay()
{
  if(document.getElementById("overlay_forceFullscreen").style.display=="block") return "forceFullscreen";
  return currentDisplayType;
}


function forceFullscreen() {
  // return; //for debug

  if (isNotFullscreen()) {
    if(!firstFullscreenRequest)
    {
      const textDiv=document.getElementById("forceFullscreen-content");
      textDiv.innerHTML="全画面表示になっていません<br><br>再び全画面表示を試みます<br><br>[Enterキーを押して進む]"
    }

    document.getElementById("overlay_forceFullscreen").style.display = "block";
    document.getElementById("jspsych-content").style.display = "none";
    document.getElementById("overlay_forceFullscreen").focus();
  }
  else
  {
    document.getElementById("overlay_forceFullscreen").style.display = "none";
    document.getElementById("jspsych-content").style.display = "block";
    document.getElementById("jspsych-content").focus();
  }
  firstFullscreenRequest=false;
}


function getCurrentFormattedTime() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0'); // Months are zero-based
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');

  return `${year}${month}${day}${hours}${minutes}${seconds}`;
}


class Trial_BrowserCheck{
  type=jsPsychBrowserCheck;

  constructor(){
  }

  features = ["width", "height", "browser", "browser_version", "mobile", "os", "fullscreen"];
}


function _getWindowSize()
{
  return {innerWidth: window.innerWidth, innerHeight: window.innerHeight,
    outerWidth: window.outerWidth, outerHeight: window.outerHeight,
    screenWidth: screen.width, screenHeight: screen.height,
    devicePixelRatio: window.devicePixelRatio};
}


class Trial_Init{
  type=jsPsychCallFunction;

  constructor(params){
    this.params=params;
  }

  func=()=>{
    document.addEventListener("keydown", (event)=>keyListener_down(event));
    document.addEventListener("keyup", (event)=>keyListener_up(event));

    let lis_fullscreen, lis_resize;
    document.addEventListener("fullscreenchange", lis_fullscreen=()=>{
      pushTimeHistory("fullscreenchange", {isFullscreen:!isNotFullscreen(), windowSize:_getWindowSize()});
      console.table("fullscreenchange", {isFullscreen:!isNotFullscreen(), windowSize:_getWindowSize()});

      forceFullscreen();
    });

    window.addEventListener("resize", lis_resize=()=>{
      pushTimeHistory("resize", {isFullscreen:!isNotFullscreen(), windowSize:_getWindowSize()});
      console.table("resize", {isFullscreen:!isNotFullscreen(), windowSize:_getWindowSize()});

      forceFullscreen();
e
      if(trialInfo_now!=null && currentDisplay()=="stim")
      {
        set_svgPos(trialInfo_now.svgScale, trialInfo_now.coordType);
      }
    });

    this.removeListeners=()=>{
      document.removeEventListener("fullscreenchange", lis_fullscreen); 
      window.removeEventListener("resize", lis_resize);}
    ;
  };

  on_finish=(data)=>{
    data.trialName="Init";
    data.sessionParam=sessionParam;
    data.params=Object.fromEntries(this.params);
    data.sessionID=this.params.get("id")+"_"+getCurrentFormattedTime();
  };
}


class Trial_Preload{
  type=jsPsychPreload;

  constructor(trialInfo){
    this.trialInfo=trialInfo
  } 

  on_load=()=>{
    trialInfo_now.copyFrom(this.trialInfo);
  };
  
  images=()=>{
    console.log("images");
    const file=this.trialInfo.paths;
    return [file,];
  };

  on_error=(f)=>{console.log("Error loading", f);};
  on_success=(f)=>{console.log("Successfully loaded", f);};

  on_finish=(data)=>{
    data.trialName="Preload";
  }
}


function updateOption(option)
{
  document.querySelector("#valueAdjustable").innerHTML=`${option}`;

  chosenOptionHistory.push(option);
  pushTimeHistory("updateOption", {option:option});
}

function increment_option(sign)
{
  let currentOption=latestOption();
  let nextOption=currentOption+sign*delta
  if(nextOption<delta){nextOption=delta;}
  if(nextOption>10000){nextOption=10000;}
  updateOption(nextOption);
}


function keyRepeat(sign, delayMS)
{
  if(!keyOn) return;
  increment_option(sign);
  if(delayMS<16) delayMS=16;
  timeout_keyRepeat=setTimeout(() =>keyRepeat(sign, delayMS*0.875), delayMS);
}


function startKeyRepeat(sign)
{
  if(keyOn) return;
  keyOn=true;

  const notice=document.getElementById("notice");
  if(notice.style.visibility=="visible")
  {
    if(trialInfo_now.blockType=="training" && sign==_trueDirection())
    {
      notice.textContent="数値を調整したら、Enterキーを押して進んでください。";
    }
    else if(trialInfo_now.blockType=="main")
    {
      notice.style.visibility="hidden";
    }
  }

  keyRepeat(sign, 256);
}


function keyListener_down(event)
{
  let key=event.key;

  let incKey, decKey;
  {
    incKey="ArrowUp";
    decKey="ArrowDown";
  }

  if(currentDisplay()=="stim")
  {
    if(trialInfo_now.blockType=="main" || trialInfo_now.blockType=="training" && (trialInfo_now.state=="Adjustable" || trialInfo_now.state=="FixedAdjustable"))
    {
      if(trialInfo_now.adjustable)
      {
        if(key==incKey)
        {
          if(!event.repeat) startKeyRepeat(+1);
        }
        else if(key==decKey)
        {
          if(!event.repeat) startKeyRepeat(-1);
        }
      }
      
      if(key=="Enter")
      {
        tryFinishTrial();
      }
    }

    else
    {
      if(key=="Enter")
      {
        finishTrial(0);
      }
    }
    
  }

  else if(currentDisplay()=="title")
  {
    if(key=="Enter")
    {
      finishTrial(0);
    }
  }

  else if(currentDisplay()=="forceFullscreen")
  {
    if(key=="Enter")
    {
      run_requestFullscreen();
    }
  }
}

function keyListener_up(event)
{
  keyOn=false;
  clearTimeout(timeout_keyRepeat);
}


function finishTrial(delayMS)
{
  pushTimeHistory("finishTrial");

  currentDisplayType=null;
  if(delayMS==0)
  {
    jsPsych.finishTrial();
  }
  else  //This does not happen
  {
    setTimeout(() => {
      jsPsych.finishTrial();
    }, delayMS);
  }
}


function _trueDirection()
{
  const initOption=chosenOptionHistory[0];
  const baseOption=100;
  const trueDirection=Math.sign(baseOption-initOption);
  return trueDirection;
}


function did_movedToTrueDirection()
{
  const trueDirection=_trueDirection();
  if(chosenOptionHistory==null || chosenOptionHistory.length<=1) return false;
  for(let i=1; i<chosenOptionHistory.length; ++i)
  {
    const diff=chosenOptionHistory[i]-chosenOptionHistory[i-1];
    if(Math.sign(diff)==trueDirection) return true;
  }
  return false;
}


function tryFinishTrial()
{
  let canFinish;
  if(trialInfo_now.blockType=="training")
  {
    canFinish=did_movedToTrueDirection();
  }
  else if(trialInfo_now.blockType=="main")
  {
    canFinish=chosenOptionHistory!=null && chosenOptionHistory.length>1;
  }
  
  pushTimeHistory("tryFinishTrial", {canFinish:canFinish});

  if(canFinish)
  {
    finishTrial(0);
  }
  else
  {
    const notice=document.getElementById("notice");
    notice.style.visibility="visible";
    shakeNotice();
  }
}


function shakeNotice()
{
  document.getElementById("notice").classList.add("shake");
}


function removeShake()
{
  document.getElementById("notice").classList.remove('shake');
}


class Trial_Title{
  type=jsPsychHtmlKeyboardResponse;

  constructor(blockType){
    this.blockType=blockType;
  }

  stimulus=()=>{
    prompt={
      "training":"図形の見え方を調べます。<br>まずは練習です。<br>全部で4問あります。",
      "main":"次から本番です。<br>解説が表示されません。<br>全部で32問あります。",
    }[this.blockType];
    return `
    <div id="container-flex"><div id="container-center">
      <div class="p">
      ${prompt}
      <br><br>
      Enterキーを押して進む。
      </div>
        
      </div>
    </div></div>
    `;}
  ;

  on_load=()=>{
    initHistory();
    
    forceFullscreen();

    currentDisplayType="title";
  };

  choices=["NO_KEYS"];

  on_finish=(data)=>{
    data.trialName="Title";
    saveHistory(data);
  };
}


function _viewBoxToWH(svgDoc)
{
  const viewBox=svgDoc.querySelector("svg").getAttribute("viewBox").split(" ");
  const svgWidth = parseFloat(viewBox[2]);
  const svgHeight = parseFloat(viewBox[3]);
  return {width:svgWidth, height:svgHeight};
}


function _swapWH(size)
{
  if(Object.hasOwn(size, "width")) //scale size
  {
    return {width:size.height, height:size.width};
  }

  if(Object.hasOwn(size, "Fixed"))
  {
    return {Fixed:_swapWH(size.Fixed), Adjustable:_swapWH(size.Adjustable)};
  }
  
}


function _scaleWH(size, scale)
{
  if(typeof scale=="number")
  {
    return {width:size.width*scale, height:size.height*scale};
  }

  //scale is an object
  if(Object.hasOwn(size, "width")) //scale size
  {
    return {width:size.width*scale.width, height:size.height*scale.height};
  }
  if(Object.hasOwn(size, "left"))  //scale pos
  {
    return {left:size.left*scale.width, top:size.top*scale.height};
  }

  if(Object.hasOwn(size, "Fixed"))
  {
    return {Fixed:_scaleWH(size.Fixed, scale), Adjustable:_scaleWH(size.Adjustable, scale)};
  }
}


function init_svgPos()
{
  const headRect=document.getElementById("head").getBoundingClientRect();
  const footRect=document.getElementById("foot").getBoundingClientRect();
  const containerRect=document.getElementById("container-flex").getBoundingClientRect();
  
  const upperScaleH=1-1/8;

  const widthUpper=containerRect.width;
  const heightUpper_full=containerRect.height-(headRect.height+footRect.height);
  const heightUpper=heightUpper_full*upperScaleH;

  const origSize_viewport={width:87.5, height: 87.5}
  const containerScale={width:origSize_viewport.width/containerRect.width, height:origSize_viewport.height/containerRect.height};

  const svgSize=structuredClone(canvSize);
  
  let scaleFull={};
  for(let coordIndex=0; coordIndex<2; ++coordIndex)
  {
    const coordType=coordType_all[coordIndex];
  
    const upper_wtoh=widthUpper/heightUpper;
    const svg_wtoh=svgSize[coordType].width/svgSize[coordType].height;
    if(upper_wtoh>svg_wtoh)
    {
      scaleFull[coordType]=heightUpper/svgSize[coordType].height;
    }
    else
    {
      scaleFull[coordType]=widthUpper/svgSize[coordType].width;
    }
  }

  const heightMarginScale=1/8;

  svgPos_allScale={}
  for(let sc=0; sc<svgScale_all.length; ++sc)
  {
    const scale=svgScale_all[sc];
    
    svgPos_allScale[scale]={Cart: null, Polar: null};
    for(let coordIndex=0; coordIndex<2; ++coordIndex)
    {
      const coordType=coordType_all[coordIndex];

      const scaleFinal=scaleFull[coordType]*scale;
      const svgSize_scaled=_scaleWH(svgSize[coordType], scaleFinal);
      
      const withMargin_full={width: widthUpper, height: svgSize[coordType].height*scaleFull[coordType]*(1+heightMarginScale)};
      
      const pos_top=(withMargin_full.height-svgSize_scaled.height)/2;
      const pos_left=(withMargin_full.width-svgSize_scaled.width)/2;
      const positions={top:pos_top, left:pos_left};

      svgPos_allScale[scale][coordType]={
        scaledSizes: _scaleWH(svgSize_scaled, containerScale),
        positions: _scaleWH(positions, containerScale),
        withMargin_full: _scaleWH(withMargin_full, containerScale),
      };
    }
  }
}


function set_svgPos(scale, coordType, flip)
{
  // if(svgPos_allScale==null) init_svgPos();
  init_svgPos();
  const svgPos=svgPos_allScale[scale][coordType];

  const svgObj=document.getElementById(`stimSVG`);
  
  const w=svgPos.scaledSizes.width;
  const h=svgPos.scaledSizes.height;
  const left=svgPos.positions.left;
  const top=svgPos.positions.top;
  
  svgObj.style.width=`${w}svw`;
  svgObj.style.height=`${h}svh`;
  svgObj.style.left=`${left}svw`;
  svgObj.style.top=`${top}svh`;
  if(flip)
  {
    svgObj.style.transform="scaleX(-1)";
  }

  const outmost=document.getElementById("stim-container-outmost");
  outmost.style.width=`${svgPos.withMargin_full.width}svw`;
  outmost.style.height=`${svgPos.withMargin_full.height}svh`;

  document.querySelectorAll(".td").forEach((elem)=>elem.style.width=`${w/2*0.96875}svw`); /* 87.5/2 = 43.75, but 43.75 does not fit in the space.*/
}


function start_animation_training()
{
  const singleDur_ms=1000;
  
  const svgObjAdjustable=document.getElementById("stimSVGAdjustable");
  const svgDocAdjustable=svgObjAdjustable.contentDocument;
  const binSize=JSON.parse(svgDocAdjustable.querySelector("#bar_height").textContent)[0].length;
  
  const elements=[];
  for(let ib=0; ib<binSize; ++ib)
  {
    const element_all=[];
    elements.push(element_all);
    for(let ift=0; ift<2; ++ift)
    {
      const stimType=["Fixed", "Adjustable"][ift];
      const svgObj=document.getElementById(`stimSVG${stimType}`);
      const svgDoc=svgObj.contentDocument;
      for(let ie=0; ie<2; ++ie)
      {
        const elementId=["s", "t"][ie];
        // const elem=svgDoc.querySelector(`#${elementId}${ib}`);
        const elem_list=svgDoc.querySelectorAll(`.${elementId}${ib}`);
        for(let i=0; i<elem_list.length; ++i)
        {
          element_all.push(elem_list[i]);
        }
      }
    }
  }
  
  // Start the animation from the first element
  animateOpacity(0, singleDur_ms, elements);
}


function animateOpacity(index, singleDur_ms, elements) {

  const binSize = elements.length;
  if (index >= binSize) index = 0;

  // Set the current element's opacity to 1
  for (let ie = 0; ie < elements[index].length; ++ie) {
    elements[index][ie].style.opacity = 1;
  }

  // After singleDur, set the opacity back to 0 and move to the next element
  setTimeout(() => {
    for (let ie = 0; ie < elements[index].length; ++ie) {
      elements[index][ie].style.opacity = 0;
    }
    animateOpacity(index + 1, singleDur_ms, elements);
  }, singleDur_ms);
}


function finalizeStimLoading(trialInfo)
{
  set_svgPos(trialInfo.svgScale, trialInfo.coordType, sessionParam.posFixed=="right");
    
  // if(trialInfo.blockType=="training")
  // {
  //   start_animation_training();
  // }

  const svgDoc=document.getElementById("stimSVG").contentDocument;
  svgDoc.addEventListener("keydown", (event)=>keyListener_down(event));
  svgDoc.addEventListener("keyup", (event)=>keyListener_up(event));

  pushTimeHistory("stimReady");
  currentDisplayType="stim";
}


function svgOnload(trialInfo)
{
  const svgObj=document.getElementById("stimSVG");
  const svgDoc=svgObj.contentDocument;

  let initOption;
  if(trialInfo.blockType=="training")
  {
    if(trialInfo.splitRatio>1)
    {
      initOption=50;
    }
    else
    {
      initOption=150;
    }
  }
  else if(trialInfo.blockType=="main")
    {
      let initOption_lower, initOption_upper;
      if(trialInfo.splitRatio>1)
      {
        initOption_lower=delta*2;
        initOption_upper=50;
      }
      else
      {
        initOption_lower=150;
        initOption_upper=200-delta*2;
      }
      initOption=jsPsych.randomization.randomInt(initOption_lower, initOption_upper);  //inclusive
      initOption=Math.round(initOption/delta)*delta;
  }
  
  if(trialInfo.blockType=="training" && (trialInfo_now.state=="Adjustable" || trialInfo_now.state=="FixedAdjustable"))
  {
    // const trueOption=JSON.parse(svgDoc.querySelector("#trueOption").textContent)[sessionParam.posFixed];
    // trialInfo_now.trueOption=trueOption;
    trialInfo_now.adjustable=true;
  }
  if(trialInfo.blockType=="main")
  {
    trialInfo_now.adjustable=true;
  }
  updateOption(initOption);

  finalizeStimLoading(trialInfo);
}


function make_coordTypeStr(coordType)
{
  const coordTypeStr={
    "Polar": "扇形",
    "Cart": "棒",
  }[coordType];
  return coordTypeStr;
}


function make_eqTypeStr(coordType, eqType)
{
  if(eqType=="Area")
  {
    return "面積";
  }

  if(coordType!=null)
  {
    const coordTypeStr={"Cart":"縦", "Polar":"半径"}[coordType];

    return `${coordTypeStr}の長さ`;
  }
  else
  {
    return "長さ";
  }
}


function make_posFixedStr(posFixed)
{
  const posFixedStr={
    "top": "上",
    "bottom": "下",
    "left": "左",
    "right": "右",
  }[posFixed];
  return posFixedStr;
}


function make_posAdjustableStr(posFixed)
{
  const posAdjustableStr={
    "top": "下",
    "bottom": "上",
    "left": "右",
    "right": "左",
  }[sessionParam.posFixed];
  return posAdjustableStr;
}


function set_byPosFixed(posFixed, valueFixed, valueAdjustable)
{
  if(posFixed=="left")
  {
    return {"left":valueFixed, "right":valueAdjustable};
  }
  else if(posFixed=="right")
  {
    return {"left":valueAdjustable, "right":valueFixed};
  }
}


class Trial_Tutorial
{
  #stimulus;

  type=jsPsychHtmlKeyboardResponse;

  constructor(trialInfo, state){
    this.trialInfo=trialInfo;
    this.state=state;

    const coordTypeStr=make_coordTypeStr(this.trialInfo.coordType);
    const eqTypeStr=make_eqTypeStr(this.trialInfo.coordType, sessionParam.eqType);

    const posFixedStr=make_posFixedStr(sessionParam.posFixed);
    const posAdjustableStr=make_posAdjustableStr(sessionParam.posFixed);

    const no_ga={"Len":"が", "Area":"の"}[sessionParam.eqType]

    let prompt=`
      次の図形をよく見てください。
      <span class="hidden vis_Shape vis_Fixed vis_Adjustable vis_FixedAdjustable">
        <br>${eqTypeStr}${no_ga}異なる${coordTypeStr}が隣り合っています。
      </span>
      <span class="hidden vis_Fixed vis_Adjustable vis_FixedAdjustable">
        <br>${posFixedStr}側の${coordTypeStr}の${eqTypeStr}を足し合わせると100になります。
      </span>
      <span class="hidden vis_Adjustable vis_FixedAdjustable">
        <br>${posAdjustableStr}側の${coordTypeStr}の${eqTypeStr}を足し合わせるといくらになるか、回答してください。
      </span>
      <span class="hidden vis_Orig vis_Shape vis_Fixed small">
        <br>[Enterキーを押して進む]
      </span>
      <span class="hidden vis_Adjustable vis_FixedAdjustable">
        <br>回答方法：上下矢印⇅キーで数値を調整<span class="small">（長押しで速く進む）</span>。Enterキーで決定。
      </span>
      `;

    const idValue=set_byPosFixed(sessionParam.posFixed, "", ' id="valueAdjustable"');
    const adjustableText=set_byPosFixed(sessionParam.posFixed, "", '↑<br>この数値を調整して回答');
    const valueInit=set_byPosFixed(sessionParam.posFixed, "100", "");
    const tdClass=set_byPosFixed(sessionParam.posFixed, " vis_Fixed vis_Adjustable vis_FixedAdjustable", " vis_Adjustable vis_FixedAdjustable");

    this.#stimulus=`
      <div id="container-flex"><div id="container-center">
        <div id="head" class="p">
          <div class="p" style="text-align:left;display:inline-block;">
            ${prompt}
          </div>
        </div>
        <div id="stim-container-outmost">
          <object type="image/svg+xml" data="${this.trialInfo.paths}" id="stimSVG"></object>
        </div>
        <div id="foot" class="p" style="visibility:hidden">
          <div class="p">
            <div>
              <span class="td${tdClass['left']}">左側</span>
              <span class="td${tdClass['right']}">右側</span>
            </div>
            <div>
              <span class="td${tdClass['left']}"${idValue['left']}>${valueInit['left']}</span>
              <span class="td${tdClass['right']}"${idValue['right']}>${valueInit['right']}</span>
            </div>
            <div>
              <span class="td${tdClass['left']}">${adjustableText['left']}</span>
              <span class="td${tdClass['right']}">${adjustableText['right']}</span>
            </div>
            <br>
            <div id="notice" class="p hidden">
              もう少し数値を上下に動かして練習してみてください。
            </div>
          </div>
        </div>
      </div></div>
    `;
  }

  stimulus=()=>this.#stimulus;

  on_load=()=>{
    trialInfo_now.state=this.state;

    console.table(this.trialInfo);

    initHistory();
    
    document.querySelectorAll(`.vis_${this.state}`).forEach((elem)=>elem.style.visibility="visible");

    document.getElementById("stimSVG").addEventListener('load', ()=>svgOnload(this.trialInfo));

    document.getElementById("notice").addEventListener('animationend', removeShake);
  };

  choices=["NO_KEYS"];

  on_finish=(data)=>{
    data.trialName="Stim";
    data.trialInfo=this.trialInfo;
    saveHistory(data);
  };
}


class Trial_Stim
{
  #stimulus;

  type=jsPsychHtmlKeyboardResponse;

  constructor(trialInfo){
    this.trialInfo=trialInfo;

    const coordTypeStr=make_coordTypeStr(this.trialInfo.coordType);
    const eqTypeStr=make_eqTypeStr(this.trialInfo.coordType, sessionParam.eqType);

    const posFixedStr=make_posFixedStr(sessionParam.posFixed);
    const posAdjustableStr=make_posAdjustableStr(sessionParam.posFixed);

    const prompt=`${posFixedStr}側の${coordTypeStr}の${eqTypeStr}を足し合わせると100になります。${posAdjustableStr}側は？`;

    let arrowStr="⇅";

    const idValueFixed="";
    const idValueAdjustable=' id="valueAdjustable"';
    const valueInitFixed="100";
    const valueInitAdjustable="";
    const idValueLeft={"left":idValueFixed, "right":idValueAdjustable}[sessionParam.posFixed];
    const idValueRight={"left":idValueAdjustable, "right":idValueFixed}[sessionParam.posFixed];
    const valueInitLeft={"left":valueInitFixed, "right":valueInitAdjustable}[sessionParam.posFixed];
    const valueInitRight={"left":valueInitAdjustable, "right":valueInitFixed}[sessionParam.posFixed];

    this.#stimulus=`
      <div id="container-flex"><div id="container-center">
        <div id="head" class="p">
          <div class="p">
            ${prompt}
          </div>
        </div>
        <div id="stim-container-outmost">
          <object type="image/svg+xml" data="${this.trialInfo.paths}" id="stimSVG"></object>
        </div>
        <div id="foot" class="p">
          <div class="p">
            <div><span class="td">左側</span><span class="td">右側</span></div>
            <div><span class="td"${idValueLeft}>${valueInitLeft}</span><span class="td"${idValueRight}>${valueInitRight}</span></div>
            <br><br>
            <span class="small">回答方法：上下矢印${arrowStr}キーで数値を調整（長押しで速く進む）。Enterキーで決定。</span>
          </div>
          <br>
          <div id="notice" class="p hidden">
            数値を調整してください。
          </div>
        </div>
      </div></div>
    `;
  }

  stimulus=()=>this.#stimulus;

  on_load=()=>{
    console.table(jsPsych.data.get().trials);
    console.table(this.trialInfo);

    initHistory();

    document.getElementById("stimSVG").addEventListener('load', ()=>svgOnload(this.trialInfo));

    document.getElementById("notice").addEventListener('animationend', removeShake);
  };

  choices=["NO_KEYS"];

  on_finish=(data)=>{
    data.trialName="Stim";
    data.trialInfo=this.trialInfo;
    saveHistory(data);
  };
}


class Trial_Save{
  type=jsPsychCallFunction;

  constructor(completed=false){
    this.completed=completed;
  }

  func=()=>{
    const data=jsPsych.data.get();
    const interactionData=jsPsych.data.getInteractionData();
    data.trials[0].interactionData=interactionData.trials;
    const completedState={"completed":this.completed};
    data.trials[0].completedState=completedState;
    const eqtid=data.trials[0].params["eqtid"];
    if(this.completed)
    {
      const sessionID=data.trials[0].sessionID;
      const path_all=[];
      completedState["path_all"]=path_all;
      
      if(sessionParam.expType=="singlePeak")
      {
        path_all.push(`./Completed_${sessionParam.expType}/${eqtid}/eqType_posFixed/${sessionParam.eqType}_${sessionParam.posFixed}/${sessionID}`);
      }
      else if(sessionParam.expType=="random")
      {
        path_all.push(`./Completed_${sessionParam.expType}/${eqtid}/Bin_PtF/Bin${sessionParam.binSize}_Ptf${sessionParam.peak_to_trough}/${sessionID}`);
        path_all.push(`./Completed_${sessionParam.expType}/${eqtid}/eqType_posFixed/Bin${sessionParam.binSize}_Ptf${sessionParam.peak_to_trough}/${sessionParam.eqType}_${sessionParam.posFixed}/${sessionID}`);
        path_all.push(`./Completed_${sessionParam.expType}/${eqtid}/SetIndex/Bin${sessionParam.binSize}_Ptf${sessionParam.peak_to_trough}/${sessionParam.eqType}_${sessionParam.posFixed}/Set${sessionParam.setIndex}`);
      }
    }
    console.table(data);
    saveData(data.json());
  };
}


class Trial_End{
  type=jsPsychHtmlKeyboardResponse;

  constructor(params, beforeunload_func, trial_init){
    this.params=params;
    this.beforeunload_func=beforeunload_func;
    this.trial_init=trial_init;
  }

  stimulus=()=>{
    return `
    <div id="container-flex"><div id="container-center">
      <div class="p">
      Enterキーを押して調査を完了
      <br><br>
      全画面表示を終了します
      </div>
    </div></div>
    `;}
  ;

  choices=["Enter"];

  on_load=()=>{ 
    this.trial_init.removeListeners();
  };
  
  on_finish=(data)=>{
    if(document.fullscreenElement!=null)
    {
      document.exitFullscreen();
    }

    window.removeEventListener('beforeunload', this.beforeunload_func);

    const params_forEnd=new URLSearchParams();
    params_forEnd.set("eqtid", this.params.get("eqtid"));
    
    // if(isInternal(this.params))
    {
      params_forEnd.set("internal", this.params.get("internal"));
      params_forEnd.set("sessionID", jsPsych.data.get().trials[0].sessionID);
      params_forEnd.set("dataLength", jsPsych.data.get().trials.length-2);
      params_forEnd.set("expType", sessionParam.expType);
      const url=`end.html?${params_forEnd.toString()}`;
      console.log(url);
      document.location.href = url;
    }

    document.getElementById("jspsych-content").innerHTML="ページ移動中<br>少しお待ち下さい";
  };
}


function parseParam(params, key, valueOption_all)
{
  const param = params.get(key);
  let value;
  if(param!=null && valueOption_all.includes(param) && isInternal(params))
  {
    value=param;
  }
  else
  {
    value=Utils.choice(valueOption_all);
  }
  return value;
}


function isInternal(params)
{
  const internal=params.has("internal") && params.get("internal")!="false";
  return internal;
}


function saveData(data){
  var xhr = new XMLHttpRequest();
  xhr.open('POST', 'saveData.php');
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(data);
  
  xhr.onload = function() {
    if (xhr.status === 200) {
      console.log("Data saved successfully");
    } else {
      console.error("Error saving data");
    }
    console.log(xhr.response)
  };
}


function factorial_shuffle_noConsecutive(factors)
{
  const stimParam_all=jsPsych.randomization.factorial(factors, 1);
  for(let i=1; i<stimParam_all.length; ++i)
  {
    if(stimParam_all[i].coordType==stimParam_all[i-1].coordType && stimParam_all[i].split==stimParam_all[i-1].split) return factorial_shuffle_noConsecutive(factors);
  }
  return stimParam_all;
}
