window.onload = function() {
    $(".txt input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".txt input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });

    $(".signbtn").on("click",function(){
        if($(".signup").css("display")=="none"){
            $(".login").hide();
            $(".signup").show();
        }else{
            $(".login").show();
            $(".signup").hide();
        }
    });

    $(".resetbtn").on("click",function(){
        $(".resetIn").show(); 
        $(".login").hide();
        $(".signup").hide();
    });

    $(".rbtn").on("click",function(){
        $(".resetIn").hide(); 
        $(".login").show();
        $(".signup").hide();
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