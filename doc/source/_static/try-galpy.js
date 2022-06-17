// Function(s) to deal with the try-galpy activation etc.
$(document).ready(function() {
    $("#activate-try-galpy-button").click(function() {
    console.log("Here");
    $(this).parent().parent().prepend('<iframe src="https://www.galpy.org/repl?code=import%20galpy%0A%23Type%20some%20code%20and%20press%20Shift%2BEnter%20to%20run" width="100%" height="525px" style="display:block;margin: 0 auto;"></iframe>')
    $(this).parent().remove();
    });
});