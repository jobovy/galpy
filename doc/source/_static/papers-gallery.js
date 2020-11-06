$(document).ready(function() {
  // Create the list of image divs with papers using galpy
  //https://d2g2jw00kypyq.cloudfront.net = https://www.galpy.org (but latter doesn't work)
  $.getJSON("https://d2g2jw00kypyq.cloudfront.net/data/papers-using-galpy.json")
    .done(function(data) {
    // First update the number of papers using galpy listed on the page
    let paperSpan= document.getElementById("span-number-of-papers-using-galpy");
    paperSpan.innerHTML= (Math.floor((Object.keys(data).length-1)/50)*50).toFixed(0);
    // randomize array to create random starting point, https://stackoverflow.com/a/56448185/10195320
      function shuffle(obj){
        // new obj to return
        let newObj = {};
        // create keys array
        var keys = Object.keys(obj);
        // randomize keys array
        keys.sort(function(a,b){return Math.random()- 0.5;});
        // save in new array
        keys.forEach(function(k) {
          newObj[k] = obj[k];
        });
        return newObj;
      }
      $.each(shuffle(data),function(key,bibentry) {
        if ( key != "_template" && bibentry.hasOwnProperty("img")) {
	    $("<div>").attr("id",`${key}-div1`).addClass("responsive").appendTo("#papers-gallery");
	    $("<div>").attr("id",`${key}-div2`).addClass("gallery").appendTo(`#${key}-div1`);
	    $("<a>").attr("id",`${key}-link`).attr("href",bibentry.url).attr("target","_blank")
		.attr("alt",`Figure from ${bibentry.title}, ${bibentry.author} (${bibentry.year}), ${bibentry.journal} ${bibentry.volume}, ${bibentry.pages}`)
		.attr("title",`Figure from ${bibentry.title}, ${bibentry.author} (${bibentry.year}), ${bibentry.journal} ${bibentry.volume}, ${bibentry.pages}`).appendTo(`#${key}-div2`);
	    $("<div>").attr("id",key).addClass("papers-gallery-item").appendTo(`#${key}-link`);
	    $("<img>").attr("data-src","https://d2g2jw00kypyq.cloudfront.net/data/paper-figs/"+bibentry.img).attr("loading","lazy").addClass("lazyload").appendTo(`#${key}`);
	    $("<div>"+`<font size="-3"><i>${bibentry.title}</i>, ${bibentry.author} (${bibentry.year}), ${bibentry.journal} ${bibentry.volume}, ${bibentry.pages}</font>`+"</div>").addClass("desc").appendTo(`#${key}-div2`);
	}
	  });
      // Lazy loading of images...
      if ('loading' in HTMLImageElement.prototype) {
	  const images = document.querySelectorAll("img.lazyload");
	  images.forEach(img => {
		  img.src = img.dataset.src;
	      });
      } else {
	  // Dynamically import the LazySizes library
	  let script = document.createElement("script");
	  script.async = true;
	  script.src =
	      "https://cdnjs.cloudflare.com/ajax/libs/lazysizes/4.1.8/lazysizes.min.js";
	  document.body.appendChild(script);
      }
	})
      .fail(function(jqxhr, textStatus, error ) {
        console.log( "Failed to load JSON gallery file");
	// Add div with warning that the gallery failed to load
	$("<div>").attr("id","papers-warning").addClass("admonition warning").appendTo("#papers-gallery");
	$("<p>Warning</p>").addClass("admonition-title").appendTo("#papers-warning");
	$("<p>Failed to load papers gallery.</p>").appendTo("#papers-warning");
	// Make sure the warning spans the page
	$("#papers-gallery").css("width","100%");
      });
});
