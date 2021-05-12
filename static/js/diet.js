window.onload = function() {
    $(".txt input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".txt input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });

    // $(".submit_btn").on("click",function(){
    //     $(".mealrec_content").show(); 
    //     $(".mealrec_input").hide();
    // });

    // $(".regenerate_btn").on("click",function(){
    //     $(".mealrec_input").show();
    //     $(".mealrec_content").hide(); 
    // });
}