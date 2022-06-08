import React, { useState } from 'react'
import "./filters.css"
import {BsIntersect} from "react-icons/bs"
import { Slider } from '@material-ui/core';
import { createTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import MenuItem from '@mui/material/MenuItem';
import ListItemText from '@mui/material/ListItemText';
import Select from '@mui/material/Select';
import Checkbox from '@mui/material/Checkbox';
import {FaVirus, FaBirthdayCake} from "react-icons/fa"
import {AiOutlineOrderedList} from "react-icons/ai"

import {IoMdImages} from "react-icons/io"


export default function Filters(props) {

  const [weightsSliderEnable, setWeightsSliderEnable] = useState(true);

  function handleWeightsOnClick(){
    setWeightsSliderEnable(!weightsSliderEnable);
  }

    /* Used to decide how many items will be shown */
    const ITEM_HEIGHT = 48;
    const ITEM_PADDING_TOP = 8;
    const NUMBER_ITMES_SHOWN = 3
    const MenuProps = {
      PaperProps: {
        style: {
          maxHeight: ITEM_HEIGHT * (NUMBER_ITMES_SHOWN+0.5) + ITEM_PADDING_TOP,
          width: 250
        }
      }
    };

    /* Used to customize the color of the slider. Seems like it cannot be done in CSS*/
    const sliderTheme = createTheme({
        overrides:{
          MuiSlider: {
            thumb:{
            color: "#555",
            },
            track: {
              color: '#555'
            },
            rail: {
              color: 'grey'
            }
          }
      }
    });

    /* List of diseases options*/
    const diseases = [
      'All',
      'Actinic keratoses and intraepithelial carcinoma',
      'Basal cell carcinoma',
      'Benign keratosis-like lesions',
      'Dermatofibroma',
      'Melanoma',
      'Melanocytic nevi',
      'Vascular lesions',
    ];

    
    function handleMaxNumberImagesOnChange(_, newValue){
      props.setMaxNumberImages(newValue)
    }
    
    function handleAgeOnChange(_, newValue){
      if (!Array.isArray(newValue)) {
        return;
      }
    
      if (newValue[0] !== props.ageInterval[0]) {
        props.setAgeInterval([Math.min(newValue[0], props.ageInterval[1]), props.ageInterval[1]]);
      } else {
        props.setAgeInterval([props.ageInterval[0], Math.max(newValue[1], props.ageInterval[0])]);
      }
    }
      function handleDiseasesChange(event) {
        const {
          target: { value },
        } = event;
    
        // On autofill we get a stringified value.
        let newValue = typeof value === 'string' ? value.split(',') : value

        if ((props.diseasesFilter).indexOf("All") > -1 && newValue.length > 0){
          newValue = diseases.filter((disease) => newValue.indexOf(disease) == -1)
        }
        // If "All" is contained in the list, we just return "All" and not all the other elements
        newValue = newValue.indexOf("All") > -1 ? ["All"] : newValue

        // If all the elements are checked but not "All" yet, we use "All" instead. The filter method computes the intersection of the two arrays
        newValue = 
        newValue.filter(function(disease) {
          return diseases.indexOf(disease) !== -1;
        }).length === diseases.length-1 ? ["All"] :newValue
  
        props.setDiseasesFilter(newValue);        
      };

    return (
        <div className="filtersContainer">

            <div className="filterContainer">
                <div className={`cliquableFilterTitleContainer ${weightsSliderEnable===true ? "active" : ""}`} onClick={handleWeightsOnClick}>
                    <BsIntersect className="filterIcon"/>
                    <div className="filterName">
                        Dimension weight
                    </div>
                </div>
                <div className = "fullWeightsContainer">
                  {weightsSliderEnable &&
                    props.distanceWeights.map((w, dim) => {
                      return <div className="weightContainer" key={"sliderDim"+dim}>
                      <div className="filterName">
                        {props.latentSpaceExplorationNames[dim]}
                      </div>
                      <ThemeProvider theme={sliderTheme}>
                      <Slider
                          aria-label={"Similarity"}
                          value={w}
                          step={0.05}
                          min={0}
                          max={1}
                          valueLabelDisplay="auto"
                          onChange={(_, newValue) => props.handleFilterWeightsChange(newValue, dim)}
                      />
                      </ThemeProvider>
                      </div>
                    })
                  }
                </div>
            </div> 


            <div className="filterContainer">
                <div className="filterTitleContainer">
                    <IoMdImages className="filterIcon"/>
                    <div className="filterName">
                        Max number of images
                    </div>
                </div>
                <ThemeProvider theme={sliderTheme}>
                <Slider
                    aria-label="Similarity"
                    value={props.maxNumberImages}
                    step={1}
                    min={1}
                    max={20}
                    valueLabelDisplay="auto"
                    onChange={handleMaxNumberImagesOnChange}
                />
                </ThemeProvider>
            </div>

            
            <div className="filterContainer">
                <div className="filterTitleContainer">
                    <FaVirus className="filterIcon"/>
                    <div className="filterName">
                        Diseases
                    </div>
                </div>
                    <Select className="diseaseSelector"
                    labelId="selectDiseaseLabel"
                    id="selectDisease"
                    multiple
                    value={props.diseasesFilter}
                    onChange={handleDiseasesChange}
                    renderValue={(selected) => selected.sort().join(", ")}
                    MenuProps={MenuProps}
                    >
                    {diseases.map((disease) => (
                        <MenuItem key={disease} value={disease}>
                        <Checkbox checked={props.diseasesFilter.indexOf(disease) > -1 || props.diseasesFilter.indexOf("All") > -1} />
                        <ListItemText primary={disease} />
                        </MenuItem>
                    ))}
                    </Select>
            </div>



            <div className="filterContainer">
                <div className="filterTitleContainer">
                    <FaBirthdayCake className="filterIcon"/>
                    <div className="filterName">
                        Age
                    </div>
                </div>
                <ThemeProvider theme={sliderTheme}>
                <Slider
                    value={props.ageInterval}
                    step={1}
                    min={0}
                    max={85}
                    valueLabelDisplay="auto"
                    onChange={handleAgeOnChange}
                />
                </ThemeProvider>
            </div>


            <div className="filterContainer">
                <div className="applyButton" onClick={props.applyOnClickHandle}>
                    Apply filters
                </div>
            </div>
            
        </div> 
    )
}
