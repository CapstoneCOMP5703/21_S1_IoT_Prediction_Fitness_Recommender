window.onload = function() {
    $(".txt input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".txt input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });
}

// $(function() {
//     $(".txt input").on("focus",function(){
//         $(this).addClass("focus");
//     });
    
//     $(".txt input").on("blur",function(){
//         if($(this).val() =="")
//         $(this).removeClass("focus");
//     });
// });