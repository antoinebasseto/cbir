import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3';

export default function ProjectionPlot(props) {

    const svgRef = useRef(null)

    useEffect(() => {
        // Setting up container
        var margin = {top: 60, right: 60, bottom: 60, left: 60},
            width = 1000 - margin.left - margin.right,
            height = 700 - margin.top - margin.bottom;
        const svg = d3.select(svgRef.current)
            .append("svg")
              .attr("width", width + margin.left + margin.right)
              .attr("height", height + margin.top + margin.bottom)
            .append("g")
              .attr("transform",
                    "translate(" + margin.left + "," + margin.top + ")");
            
        // Setting up scaling
        let xMin = d3.min(props.data, (d) => d.x);
        let xMax = d3.max(props.data, (d) => d.x);
        let yMin = d3.min(props.data, (d) => d.y);
        let yMax = d3.max(props.data, (d) => d.y);
        const xScale = d3.scaleLinear()
            .domain([xMin, xMax])
            .range([0, width]);
        const yScale = d3.scaleLinear()
            .domain([yMin, yMax])
            .range([height, 0]);

        // Setting up axis
        const xAxis = d3.axisBottom(xScale)
        const yAxis = d3.axisLeft(yScale)
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);
        svg.append('g')
            .call(yAxis)

        // Setting up axis labeling
        svg.append("text")
            .attr("text-anchor", "end")
            .attr("x", width/2 + margin.left)
            .attr("y", height + margin.bottom - 20)
            .text("UMAP1");

        svg.append("text")
            .attr("text-anchor", "end")
            .attr("transform", "rotate(-90)")
            .attr("y", -margin.left + 20)
            .attr("x", -margin.top - height/2)
            .text("UMAP2")

        // Setting up tooltips
        const tooltip = d3.select('#projectionPlotContainer')
            .append('div')
            .style("opacity", 0.7)
            .style('visibility','visible')
            .style('position','absolute')
            .attr("class", "tooltip")
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "1px")
            .style("border-radius", "5px")
            .style("padding", "10px")
        
        const mouseover = function(event, d) {
            tooltip
                .style("border-color", color(d.dx))
                .style('visibility','visible')
                .text(`diagnosis: ${d.dx}
                localization: ${d.localization}
                age: ${d.age}
                sex: ${d.sex}`)
        }
        
        const mousemove = function(event, d) {
            tooltip
                .style('top', (event.pageY+25) + 'px')
                .style('left', (event.pageX+25) + 'px')
        }
        
        const mouseleave = function(event, d) {
            tooltip
                .style('visibility','hidden')
        }

        // Setting up class colours
        var keys = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

        var color = d3.scaleOrdinal()
            .domain(keys)
            .range(d3.schemeSet2);
    
        // Setting up the legend
        svg.selectAll("legendDots")
            .data(keys)
            .enter()
            .append("circle")
            .attr("cx", 25)
            .attr("cy", function(d,i) {return i*25})
            .attr("r", 7)
            .style("fill", function(d) {return color(d)})

         svg.selectAll("legendLabels")
            .data(keys)
            .enter()
            .append("text")
              .attr("x", 45)
              .attr("y", function(d,i) {return i*25})
              .style("fill", function(d) {return color(d)})
              .text(function(d){return d})
              .attr("text-anchor", "left")
              .style("alignment-baseline", "middle")
        
        // Setting up svg data
        svg.append('g')
            .selectAll()
            .data(props.data)
            .enter()
            .append('circle')
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y))
                .attr('r', 2)
                .style("fill", function(d) {return color(d.dx)} )
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)

        // Setting up uploaded data
        svg.append('g')
            .selectAll()
            .data(props.uploadedData)
            .enter()
            .append('circle')
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y))
                .attr('r', 5)
                .style("fill", "black")
    }, [props.data, props.uploadedData])

    return (
        <div id="projectionPlotContainer" className="projectionPlotContainer">
            <svg 
                className="projectionPlotSVG" 
                width={1000}
                height={700}
                ref={svgRef}
            />
        </div>
    )
}