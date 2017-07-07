var copyCode = {
    init: function() {
	$('.highlight').each(function() {
		$(this).prepend('<button class="copy-btn" data-clipboard-snippet></button>');
	    });
	var clipboardSnippets = new Clipboard('[data-clipboard-snippet]',{
		target: function(trigger) {
		    return trigger.nextElementSibling;
		}
	    });
	clipboardSnippets.on('success', function(e) {
		e.clearSelection();
		copyCode.showTooltip(e.trigger, 'Copied!');
	    });
	clipboardSnippets.on('error', function(e) {
		copyCode.showTooltip(e.trigger, copyCode.fallbackMessage());
	    });

	$('.copy-btn').each(function() {
		$(this).mouseenter(function() {
			copyCode.showTooltip(this, 'Copy to clipboard');
		    });
		$(this).mouseleave(function() {
			$(this).removeAttr('aria-label');
			$(this).removeClass('tooltipped tooltipped-sw');
		    });
	    });
    },

    showTooltip: function(elem, msg) {
	$(elem).addClass('tooltipped tooltipped-sw');
	$(elem).attr('aria-label', msg);
    },

    fallbackMessage: function() {
	var actionMsg = '';
	if (/iPhone|iPad/i.test(navigator.userAgent)) {
	    actionMsg = 'No support :(';
	} else if (/Mac/i.test(navigator.userAgent)) {
	    actionMsg = 'Press ⌘-C to copy';
	} else {
	    actionMsg = 'Press Ctrl-C to copy';
	}
	return actionMsg;
    }
};

$(document).ready(copyCode.init);
