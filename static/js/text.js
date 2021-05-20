window.onload = function() {
    // load_data();
    $(".txt input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".txt input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });

    $(".eat_calorie input").on("focus",function(){
        $(this).addClass("focus");
    });
    
    $(".eat_calorie input").on("blur",function(){
        if($(this).val() =="")
        $(this).removeClass("focus");
    });
    
    // $(".signbtn").on("click",function(){
    //     if($(".signup").css("display")=="none"){
    //         $(".login").hide();
    //         $(".signup").show();
    //     }else{
    //         $(".login").show();
    //         $(".signup").hide();
    //     }
    // });

    // $(".resetbtn").on("click",function(){
    //     $(".resetIn").show(); 
    //     $(".login").hide();
    //     $(".signup").hide();
    // });

    // $(".rbtn").on("click",function(){
    //     $(".resetIn").hide(); 
    //     $(".login").show();
    //     $(".signup").hide();
    // });
}

// function validateForm()
// {
//   var x=document.forms["workrec-input"]["calories"].value;
//   if (x==null || x=="")
//   {
//     alert("Please input a valid number.");
//     return false;
//   }
// }
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