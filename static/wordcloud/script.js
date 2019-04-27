
/*  ======================= SETUP ======================= */
var config = {
    trace: true,
    spiralResolution: 1, //Lower = better resolution
    spiralLimit: 360 * 5,
    lineHeight: 0.8,
    xWordPadding: 0,
    yWordPadding: 3,
    font: "sans-serif"
}

var arr = data;

//console.log(typeof (arr))


var words = [];
var a = arr.split(',')
for(i=0;i<a.length;i++) {
    var w = a[i].split(':')
    var st = w[0]
    var fr = w[1]
    var word = ''
    for(var j =0; j<st.length;j++){
        if(st[j]==' ' || st[j]  == '{' || st[j] == "'" || st[j] == "'" || st[j]=='}'){}
        else
            word+=st[j];

    }
    var freq = ''
     for(var j =0; j<fr.length;j++){
        if(fr[j]==' ' || fr[j]  == '{' || fr[j] == "'" || fr[j] == "'" || fr[j]=='}'){}
        else
            freq+=fr[j];

    }

     freq = parseFloat(freq)
    words.push({word: word, freq: freq});
}

// var words = Object.values(arr).map(function(key) {
//     console.log(key)
//     return {
//
//         word: key,
//         freq: key
//     }
// })


words.sort(function(a, b) {
    return -1 * (a.freq - b.freq);
});

var cloud = document.getElementById("word-cloud");
cloud.style.position = "relative";
cloud.style.fontFamily = config.font;

var traceCanvas = document.createElement("canvas");
traceCanvas.width = cloud.offsetWidth;
traceCanvas.height = cloud.offsetHeight;
var traceCanvasCtx = traceCanvas.getContext("2d");
cloud.appendChild(traceCanvas);

var startPoint = {
    x: cloud.offsetWidth / 2,
    y: cloud.offsetHeight / 2
};

var wordsDown = [];
/* ======================= END SETUP ======================= */





/* =======================  PLACEMENT FUNCTIONS =======================  */
function createWordObject(word, freq) {
    var wordContainer = document.createElement("div");
    wordContainer.style.position = "absolute";
    wordContainer.style.fontSize = freq + 10 + "px";
    wordContainer.style.lineHeight = config.lineHeight;
//    wordContainer.style.transform = "translateX(-50%) translateY(-50%)"
    switch (word){
         case "Cognition": wordContainer.style.color = "#4286f4";
             break;
         case "Performance": wordContainer.style.color = "#ff9966";
             break;
         case "Coordination": wordContainer.style.color = "#cca610";
             break;
         case "Management": wordContainer.style.color = "#08b20e";
             break;
         case "Control": wordContainer.style.color = "#ed76b3";
             break;
         case "Procedural&Team": wordContainer.style.color = "#ed2d43";
             break;
         case "Communication": wordContainer.style.color = "#e0682c";
             break;
         case "Information": wordContainer.style.color = "#d0d831";
             break;
         case "Planning": wordContainer.style.color = "#12914f";
             break;
         case "Resources": wordContainer.style.color = "#34d3e5";
             break;
         case "Team&Organization": wordContainer.style.color = "#5c6de0";
             break;
         case "Org-Coordination": wordContainer.style.color = "#b047e5";
             break;
         default: wordContainer.style.color = "#fc5dc7";

    }
    wordContainer.appendChild(document.createTextNode(word));

    return wordContainer;
}

function placeWord(word, x, y) {

    cloud.appendChild(word);
    word.style.left = x - word.offsetWidth/2 + "px";
    word.style.top = y - word.offsetHeight/2 + "px";

    wordsDown.push(word.getBoundingClientRect());
}

function trace(x, y) {
//     traceCanvasCtx.lineTo(x, y);
//     traceCanvasCtx.stroke();
    traceCanvasCtx.fillRect(x, y, 1, 1);
}

function spiral(i, callback) {
    angle = config.spiralResolution * i;
    x = (1 + angle) * Math.cos(angle);
    y = (1 + angle) * Math.sin(angle);
    return callback ? callback() : null;
}

function intersect(word, x, y) {
    cloud.appendChild(word);

    word.style.left = x - word.offsetWidth/2 + "px";
    word.style.top = y - word.offsetHeight/2 + "px";

    var currentWord = word.getBoundingClientRect();

    cloud.removeChild(word);

    for(var i = 0; i < wordsDown.length; i+=1){
        var comparisonWord = wordsDown[i];

        if(!(currentWord.right + config.xWordPadding < comparisonWord.left - config.xWordPadding ||
             currentWord.left - config.xWordPadding > comparisonWord.right + config.wXordPadding ||
             currentWord.bottom + config.yWordPadding < comparisonWord.top - config.yWordPadding ||
             currentWord.top - config.yWordPadding > comparisonWord.bottom + config.yWordPadding)){

            return true;
        }
    }

    return false;
}
/* =======================  END PLACEMENT FUNCTIONS =======================  */





/* =======================  LETS GO! =======================  */
(function placeWords() {
    for (var i = 0; i < words.length; i += 1) {

        var word = createWordObject(words[i].word, words[i].freq);

        for (var j = 0; j < config.spiralLimit; j++) {
            //If the spiral function returns true, we've placed the word down and can break from the j loop
            if (spiral(j, function() {
                    if (!intersect(word, startPoint.x + x, startPoint.y + y)) {
                        placeWord(word, startPoint.x + x, startPoint.y + y);
                        return true;
                    }
                })) {
                break;
            }
        }
    }
})();
/* ======================= WHEW. THAT WAS FUN. We should do that again sometime ... ======================= */



/* =======================  Draw the placement spiral if trace lines is on ======================= */
(function traceSpiral() {

    traceCanvasCtx.beginPath();

    if (config.trace) {
        var frame = 1;

        function animate() {
            spiral(frame, function() {
                trace(startPoint.x + x, startPoint.y + y);
            });

            frame += 1;

            if (frame < config.spiralLimit) {
                window.requestAnimationFrame(animate);
            }
        }

        animate();
    }
})();