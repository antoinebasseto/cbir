import React, { useEffect, useRef } from 'react'
import * as d3 from 'd3';

export default function ProjectionPlot(props) {

    const svgRef = useRef(null)

    useEffect(() => {
        // Setting up container
        var margin = {top: 10, right: 30, bottom: 30, left: 60},
            w = 460 - margin.left - margin.right,
            h = 400 - margin.top - margin.bottom;
        const svg = d3.select(svgRef.current)
            .style('overflow', 'visible')
            .append("svg")
              .attr("width", w + margin.left + margin.right)
              .attr("height", h + margin.top + margin.bottom)
              .style('overflow', 'visible')
            .append("g")
              .attr("transform",
                    "translate(" + margin.left + "," + margin.top + ")");
            
            // .attr('width', w)
            // .attr('height', h)
            
        // Setting up scaling
        let xMin = d3.min(props.data, (d) => d.x);
        let xMax = d3.max(props.data, (d) => d.x);
        let yMin = d3.min(props.data, (d) => d.y);
        let yMax = d3.max(props.data, (d) => d.y);
        const xScale = d3.scaleLinear()
            .domain([0, 10])
            .range([[0, w]]);
        const yScale = d3.scaleLinear()
            .domain([0, 10])
            .range([[h, 0]]);

        // Setting up axis
        const xAxis = d3.axisBottom(xScale)
        const yAxis = d3.axisLeft(yScale)
        svg.append("g")
            .attr("transform", "translate(0," + h + ")")
            .call(xAxis);
        svg.append('g')
            .call(yAxis)

        // Setting up axis labeling
        svg.append('text')
            .attr('x', w/2)
            .attr('y', h+50)
            .text('UMAP 1')
        svg.append('text')
            .attr('x', -50)
            .attr('y', h/2)
            .text('UMAP 2')

        
        // Setting up svg data
        svg.append('g')
            .selectAll()
            .data(props.data)
            .enter()
            .append('circle')
                .attr('cx', d => {
                    console.log(xScale(0))
                    return xScale(d.x)})
                .attr('cy', d => yScale(d.y))
                .attr('r', 2)
                .style('fill', '#69b3a2')
        
    }, [props.data, svgRef.current])

    return (
        <svg
            ref={svgRef}
        />
    )
    
    // (
    //     <div className='projectionPlot'>
    //         <div className='projectionPlotWrapper'>
    //             <svg
    //                 className='d3-scatter-plot'
    //                 ref={svgRef}
    //             />
    //         </div>
    //     </div>
    // )
}