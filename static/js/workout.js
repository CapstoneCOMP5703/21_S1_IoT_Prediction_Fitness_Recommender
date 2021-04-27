window.onload = function() {
    $(".txt input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".txt input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });

    $(".submit_btn").on("click",function(){
        $(".workrec_content").show(); 
        $(".workrec-input").hide();
    });

    $(".regenerate_btn").on("click",function(){
        $(".workrec-input").show();
        $(".workrec_content").hide(); 
    });


}

// function load_data(){
//     var theme=localStorage.getItem("username");
//     if(theme==null||theme==""){
//        $("#cue").show(); 
//         $("#uname").html('');
//     }else{
//         $("#cue").hide();  
//         $("#uname").html(theme);
//     }
// }