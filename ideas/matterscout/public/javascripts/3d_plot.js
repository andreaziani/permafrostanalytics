var path = "/data/normal/seismic_data.csv";

var id = $("#get-event").val()


x = [];
y = [];
z = [];

if (id === 1) {

}


    Plotly.d3.csv(path, function(err, rows){
    function unpack(rows, key) {
        return rows.map(function(row)
        { return row[key]; });}

    var trace1 = {
        x:[2], y: [1], z: [4],
        mode: 'markers',
        marker: {
            size: 1,
            line: {
                color: 'rgba(217, 217, 217, 0.14)',
                width: 0.5},
            opacity: 0.8},
        type: 'scatter3d'
    };

    var trace2 = {
        x:[3], y: [2], z: [2],
        mode: 'markers',
        marker: {
            color: 'rgb(127, 127, 127)',
            size: 12,
            symbol: 'circle',
            line: {
                color: 'rgb(204, 204, 204)',
                width: 1},
            opacity: 0.8},
        type: 'scatter3d'};

    var data = [trace1, trace2];
    var layout = {margin: {
            l: 0,
            r: 0,
            b: 0,
            t: 0
        }};
    Plotly.newPlot('3d-div', data, layout);
});