import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3';

export default function ProjectionPlot(props) {

    const svgRef = useRef(null)

    useEffect(() => {
        // Setting up container
        var margin = {top: 10, right: 30, bottom: 30, left: 60},
            width = 460 - margin.left - margin.right,
            height = 400 - margin.top - margin.bottom;
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
            .attr("y", height + margin.top + 20)
            .text("UMAP1");

        svg.append("text")
            .attr("text-anchor", "end")
            .attr("transform", "rotate(-90)")
            .attr("y", -margin.left + 20)
            .attr("x", -margin.top - height/2 + 20)
            .text("UMAP2")

        // Setting up tooltips
        const tooltip = d3.select('#projectionPlotContainer')
            .append('div')
            .style('visibility','visible')
            .style('position','absolute')
        
        const mouseover = function(event, d) {
            tooltip
                .style('visibility','visible')
                .text(`age: ${d.age}`)
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
        
        // Setting up svg data
        svg.append('g')
            .selectAll()
            .data(props.data)
            .enter()
            .append('circle')
                .attr('cx', d => xScale(d.x))
                .attr('cy', d => yScale(d.y))
                .attr('r', 10)
                .style('fill', '#69b3a2')
            .on("mouseover", mouseover)
            .on("mousemove", mousemove)
            .on("mouseleave", mouseleave)
    }, [props.data, svgRef.current])

    return (
        <div id="projectionPlotContainer" className="projectionPlotContainer">
            <svg 
                className="projectionPlotSVG" 
                width={460} 
                height={400}
                ref={svgRef}
            />
        </div>
    )
}