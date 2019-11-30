var array_tags = [];
var counter = 0;


$(document).ready(function () {
    $("#tagbutton").on("click",function (e) {
        e.preventDefault();
        let inserted_tag = $('#tag-form').val();
        let tags = $('#labels').html();

        if (counter===0) {
            $("#submitReport").css("display", "block");
        }

        new_tag = '<span id="tag-'+counter+'" class="uk-label tag-layout"><span uk-icon="close" style="margin-right: 5px"></span>'+inserted_tag+'</span>';
        array_tags.push(counter);
        counter+=1;
        $('#labels').html(tags+new_tag);
        $('#tag-form').val("");

    });

    $(".tag-layout").on("click",function (event) {
        $(event.target.id).css("display", "none");
        console.log("clicked");
    })

});