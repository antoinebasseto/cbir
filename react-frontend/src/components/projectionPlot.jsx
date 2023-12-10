import { useEffect, useRef } from 'react'
import {queryImage} from "../backend/BackendQueryEngine";
import * as d3 from 'd3';

export default function ProjectionPlot(props) {

    const svgRef = useRef(null)

    //if (props.demo === "skin") {
        useEffect(() => {
            // Get parent size
            var width = d3.select("#projectionPlotContainer").node().getBoundingClientRect().width
            var height = d3.select("#projectionPlotContainer").node().getBoundingClientRect().height

            d3.select(svgRef.current).selectAll("*").remove()

            const svg = d3.select(svgRef.current)
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                
            // Setting up scaling
            let xMin = d3.min(props.data, (d) => d.umap1);
            let xMax = d3.max(props.data, (d) => d.umap1);
            let yMin = d3.min(props.data, (d) => d.umap2);
            let yMax = d3.max(props.data, (d) => d.umap2);

            // if (props.uploadedData) {
            //     xMin = d3.min(xMin, props.uploadedData.umap1)
            //     xMax = d3.max(xMax, props.uploadedData.umap1)

            //     yMin = d3.min(yMin, props.uploadedData.umap2)
            //     yMax = d3.max(yMax, props.uploadedData.umap2)
            // }

            let xMargin = (xMax - xMin) / 25
            let yMargin = (yMax - yMin) / 25
            const xScale = d3.scaleLinear()
                .domain([xMin - xMargin, xMax + xMargin])
                .range([0, width]);
            const yScale = d3.scaleLinear()
                .domain([yMin - yMargin, yMax + yMargin])
                .range([height, 0]);

            // Setting up label mappings
            const keys = (props.uploadedData.length > 0 || props.similarImages.length > 0) 
                ? props.labels.concat(["uploaded image", "similar image"])
                : props.labels
            
            if (props.demo === 'skin') {
                var label = d3.scaleOrdinal()
                    .domain(keys) //'akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'
                    .range(["Actinic keratoses and intraepithelial carcinoma", 
                            "Basal cell carcinoma",
                            "Benign keratosis-like lesions",
                            "Dermatofibroma",
                            "Melanoma",
                            "Melanocytic nevi",
                            "Vascular lesions",
                            "Uploaded image",
                            "Similar image"])
            } else if (props.demo === 'mnist') {
                var label = d3.scaleOrdinal()
                    .domain(keys)
                    .range(["0", 
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                            "7",
                            "8",
                            "9",
                            "Uploaded image",
                            "Similar image"])
            }

            if (props.demo === 'skin') {
                var color = d3.scaleOrdinal()
                    .domain(["Actinic keratoses and intraepithelial carcinoma", 
                        "Basal cell carcinoma",
                        "Benign keratosis-like lesions",
                        "Dermatofibroma",
                        "Melanoma",
                        "Melanocytic nevi",
                        "Vascular lesions",
                        "Uploaded image",
                        "Similar image"])
                    .range(d3.schemeSet2.slice(0,7).concat(["black", "black"]))
            } else if (props.demo === 'mnist') {
                var color = d3.scaleOrdinal()
                    .domain(["0", 
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "Uploaded image",
                        "Similar image"])
                    .range(d3.schemeSet3.slice(0,10).concat(["black", "black"]))
            }

            if (props.demo === 'skin') {
                var colorKeys = d3.scaleOrdinal()
                    .domain(keys)
                    .range(d3.schemeSet2.slice(0,7).concat(["black", "black"]))
            } else if (props.demo === 'mnist') {
                var colorKeys = d3.scaleOrdinal()
                    .domain(keys)
                    .range(d3.schemeSet3.slice(0,10).concat(["black", "black"]))
            }

            if (props.demo === 'skin') {
                var symbol = d3.scaleOrdinal()
                    .domain(keys)
                    .range([d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolSquare, d3.symbolCross])
            } else if (props.demo === 'mnist') {
                var symbol = d3.scaleOrdinal()
                    .domain(keys)
                    .range([d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolCircle, d3.symbolSquare, d3.symbolCross])
            }

            // Setting up tooltips
            const tooltip = d3.select('#projectionPlotContainer')
                .append('div')
                .style("opacity", 1)
                .style('visibility','hidden')
                .style('position','absolute')
                .attr("class", "tooltip")
                .style("background-color", "white")
                .style("border", "solid")
                .style("border-width", "1px")
                .style("border-radius", "5px")
                .style("padding", "10px")

            if (props.demo === 'skin') {
                var mouseover = function(event, d) {
                    tooltip
                        .style("border-color", colorKeys(d.label))
                        .style('visibility','visible')
                        .html(`<b>diagnosis:</b> ${label(d.label)} <br/>
                        <b>confirmation:</b> ${d.dx_type} <br/>
                        <b>localization:</b> ${d.localization} <br/>
                        <b>age:</b> ${d.age} <br/> 
                        <b>sex:</b> ${d.sex} <br/>
                        <img width="100" src=${queryImage(props.demo, d.image_id)}><img/>`)
                }
            } else if (props.demo === 'mnist') {
                var mouseover = function(event, d) {
                    tooltip
                        .style("border-color", color(d.label))
                        .style('visibility','visible')
                        .html(`<b>${d.label}</b> <br/>
                        <img width="100" src=${queryImage(props.demo, d.image_id)}><img/>`)
                }
            }
            
            const mousemove = function(event, d) {
                tooltip
                    .style('top', (event.pageY-60) + 'px')
                    .style('left', (event.pageX+25) + 'px')
            }
            
            const mouseleave = function(event, d) {
                tooltip
                    .style('visibility','hidden')
            }

            // Setting up clicking on data point
            const click = function(event, d) {
                props.handleClickOnDataPoint(d)
            }

            // Setting up data
            svg.append('g')
                .selectAll('points')
                .data(props.data)
                .enter()
                .append("path")
                    .attr("d", d3.symbol()
                        .size(20)
                        .type(d3.symbolCircle)
                    )
                    .attr("transform", function(d) { return "translate(" + xScale(d.umap1) + "," + yScale(d.umap2) + ")"; })
                    .style("opacity", function(d) {
                        if (props.uploadedData.length === 0 && props.similarImages.length === 0) {return 0.5}
                        else {return 0.5}
                    })
                    .style("fill", function(d) {return colorKeys(d.label)})
                .on("mouseover", mouseover)
                .on("mousemove", mousemove)
                .on("mouseleave", mouseleave)
                .on("click", click)


            // Setting up exploration data line
            svg.append("path")
                .datum(props.explorationData[props.dimensionExplorationProjectionFocused])
                    .attr("fill", "none")
                    .attr("stroke", "black")
                    .attr("stroke-width", 1.5)
                    .attr("d", d3.line()
                        .x(function(d) { return xScale(d.umap1) })
                        .y(function(d) { return yScale(d.umap2) })
                    )
                

            // Setting up exploration data points
            svg.append('g')
                .selectAll('explorationPoints')
                .data(props.explorationData[props.dimensionExplorationProjectionFocused])
                .enter()
                .append("path")
                    .attr("d", d3.symbol()
                        .size(20)
                        .type(d3.symbolCircle)
                    )
                    .attr("transform", function(d) { return "translate(" + xScale(d.umap1) + "," + yScale(d.umap2) + ")"; })
                    .style("opacity", 1)
                    .style("stroke", "black")
                    .style("fill", function(d, i) { 
                        if (props.rolloutClustering.length > 0) {
                            return colorKeys(props.rolloutClustering[props.dimensionExplorationProjectionFocused][i]) 
                        } else {
                            return "gray"
                        }
                    })
                

            // Setting up uploaded data
            svg.append('g')
                .selectAll('uploadedPoint')
                .data(props.uploadedData)
                .enter()
                .append("path")
                    .attr("d", d3.symbol()
                        .size(100)
                        .type(d3.symbolSquare)
                    )
                    .attr("transform", function(d) { return "translate(" + xScale(d.umap1) + "," + yScale(d.umap2) + ")"; })
                    .style("opacity", 1)
                    .style("fill", colorKeys("uploaded image"))
                    .style("stroke", "white")
                    

            // Setting up similar images data
            svg.append('g')
                .selectAll('similarPoints')
                .data(props.similarImages)
                .enter()
                .append("path")
                    .attr("d", d3.symbol()
                        .size(60)
                        .type(d3.symbolCross)
                    )
                    .attr("transform", function(d) { return "translate(" + xScale(d.umap1) + "," + yScale(d.umap2) + ")"; })
                    .style("opacity", 1)
                    .style("fill", function(d) {return color(String(d.label))})
                    .style("stroke", "black")
                .on("mouseover", mouseover)
                .on("mousemove", mousemove)
                .on("mouseleave", mouseleave)


            // Setting up legend
            svg.selectAll("legendDots")
                .data(keys)
                .enter()
                .append("path")
                    .attr("d", d3.symbol()
                        .size(function(d) {
                            if (symbol(d) === d3.symbolCross) {return 95}
                            else {return 125}
                        })
                        .type(function(d) {return symbol(d)})
                    )
                    .attr("transform", function(d,i) {return "translate(25," + (height - 30 - (11-i)*25)  + ")";})
                    .style("fill", function(d) {return colorKeys(d)})

            svg.selectAll("legendLabels")
                .data(keys)
                .enter()
                .append("text")
                .attr("x", 45)
                .attr("y", function(d,i) {return height - 30 - (11-i)*25})
                .style("fill", function(d) {return colorKeys(d)})
                .text(function(d) {return label(d)})
                .attr("text-anchor", "left")
                .style("alignment-baseline", "middle")
        }, [props.data, props.uploadedData, props.explorationData, props.similarImages, props.rolloutClustering, props.dimensionExplorationProjectionFocused])

    return (
        <div 
            id="projectionPlotContainer"
            className="w-full h-full shrink-0">
            <svg 
                className="w-full h-full"
                ref={svgRef}
            />
        </div>
    )
}