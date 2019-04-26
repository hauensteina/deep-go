
/* Various js utility funcs
   AHN, Apr 2019
 */

'use strict'

//=====================
class AhauxUtils
{
  // We need d3 and jquery
  //-------------------------
  constructor( d3, $) {
    this.d3 = d3
    this.$ = $
  } // contructor()

  //----------------------------
  //--- D3 graphics routines ---
  //----------------------------

  // Simple line chart using d3.
  // container is a string like '#some_div_id'.
  // data looks like [[x_0,y_0], ... ] .
  // xlim and ylim are pairs like [x_min,x_max] .
  //----------------------------------------------
  plot_line( container, data, xlim, ylim, color) {
    color = color || 'steelblue'
    var [d3,$] = [this.d3, this.$]
    var C = d3.select( container)
    $(container).html('')
    var w  = $(container).width()
    var h = $(container).height()

    var margin = {top: 50, right: 50, bottom: 50, left: 50}
      ,width = w - margin.left - margin.right
      ,height = h - margin.top - margin.bottom

    var scale_x = d3.scaleLinear()
      .domain([xlim[0], xlim[1]]) // input
      .range([0, width]) // output

    var scale_y = d3.scaleLinear()
      .domain([ylim[0], ylim[1]]) // input
      .range([height, 0]) // output

    var line = d3.line()
      .x(function(d, i) {
        return scale_x( d[0]) }) // set the x values for the line generator
      .y(function(d, i) {
        return scale_y( d[1]) }) // set the y values for the line generator

    // Add the SVG to the container, with margins
    var svg = C.append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')

    // Add x axis
    svg.append('g')
      .attr('class', 'x axis')
      .attr('transform', 'translate(0,' + height + ')')
      .call(d3.axisBottom(scale_x)) // run axisBottom on the g thingy

    // Add y axis
    svg.append('g')
      .attr('class', 'y axis')
      .call(d3.axisLeft(scale_y)) // run axisLeft on the g thingy

    // Draw the line
    svg.append('path')
      .datum(data) // Binds data to the line
      .attr('style', 'fill:none;stroke:' + color + ';stroke-width:3')
      .attr('d', line) // Call the line generator

  } // plot_line()

  // Barchart.
  // container is a string like '#some_div_id'.
  // data looks like [[x_0,y_0], ... ] .
  // ylim is a positive float.
  //----------------------------------------------
  barchart( container, data, ylim, color) {
    color = color || 'steelblue'
    var [d3,$] = [this.d3, this.$]
    var C = d3.select( container)
    $(container).html('')
    var w  = $(container).width()
    var h = $(container).height()

    var margin = {top: 20, right: 20, bottom: 70, left: 40}
      ,width = w - margin.left - margin.right
      ,height = h - margin.top - margin.bottom

    var svg = C.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

    var scale_x = d3.scaleBand()
      .domain( data.map( function(d) { return d[0] }))
      .rangeRound( [0, width])
      .padding( 0.05)

    var scale_y = d3.scaleLinear()
      .domain( [0, ylim])
      .range( [height, 0])

    var xAxis = d3.axisBottom(scale_x)
      .tickFormat( d3.format( '.3f'))

    var yAxis = d3.axisLeft(scale_y)

    svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
      .selectAll("text")
      .style("text-anchor", "end")
      .attr("dx", "-.8em")
      .attr("dy", "-.55em")
      .attr("transform", "rotate(-90)" );

    svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)

    svg.selectAll("bar")
      .data(data)
      .enter().append("rect")
      .style("fill", color)
      .attr("x", function(d) { return scale_x( d[0]) })
      .attr("width", scale_x.bandwidth())
      .attr("y", function(d) { return scale_y( d[1]) })
      .attr("height", function(d) { return height - scale_y( d[1]) })

  } // barchart()

  //-----------------
  //--- API stuff ---
  //-----------------

  // Hit any endpoint and call completion with result
  //---------------------------------------------------
  hit_endpoint( url, args, completion) {
    if (args.constructor.name == 'File') { // Telling api to do something with a file
      var myfile = args
      //debugger
      var data = new FormData()
      data.append( 'file',myfile)
      fetch( url,
        {
          method: 'POST',
          body: data
        }).then( (resp) => {
          resp.json().then( (resp) => { completion( resp) }) }
        ).catch(
          (error) => {
            console.log( error)
          }
        )
    } // if file
    else { // Not a file, regular api call
      fetch( url,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify( args)
        }
      ).then( (resp) => {
        resp.json().then( (resp) => { completion( resp) }) }
      ).catch(
        (error) => {
          console.log( error)
        }
      )
    } // else
  } // hit_endpoint()

} // class AhauxUtils
