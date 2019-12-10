// Sourced from: https://github.com/ianmcloughlin/random-web-app/blob/master/static/generate.js 
// Wait until the DOM is ready.
$(document).ready(function(e) {
  
    // Add a click handler to the submit button.
    $("#submitButton").click(function(e) {
      
      // Prevent the form actually submitting.
      e.preventDefault();
      
      // Get the number of numbers to generate.
      noofnos = $("#noofnos").val();
      
      // Send AJAX request for new numbers.
      $.post("/random", {"noofnos": noofnos}, function(data){
        
        // Update the text area with the numbers.
        $("#randomNumbers").text(data.randomNos.join("\n"));
      
      });
    
    });
  
  });