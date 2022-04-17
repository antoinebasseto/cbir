import React, { useState } from 'react'
import "./filters.css"
import {BsIntersect} from "react-icons/bs"
import { Slider } from '@material-ui/core';
import { createTheme } from '@material-ui/core/styles';
import { ThemeProvider } from '@material-ui/styles';
import MenuItem from '@mui/material/MenuItem';
import ListItemText from '@mui/material/ListItemText';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import Checkbox from '@mui/material/Checkbox';
import {FaVirus} from "react-icons/fa"
import {AiOutlineOrderedList} from "react-icons/ai"

import {IoMdImages} from "react-icons/io"


export default function Filters(props) {

    {/* Used to decide how many items will be shown */}
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

    {/* Used to customize the color of the slider. Seems like it cannot be done in CSS*/}
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

    {/* List of diseases options*/}
    const diseases = [
        'All',
        'Atelectasis',
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Effusion',
        'Emphysema',
        'Fibrosis',
        'Hernia',
        'Infiltration',
        'Mass',
        'Nodule',
        'No Finding',
        'Pleural Thickening',
        'Pneumonia',
        'Pneumothorax',
      ];



    function handleSimilarityOnChange(_, newValue){
        props.setSimilarityThreshold(newValue)
    }
    
    function handleMaxNumberImagesOnChange(_, newValue){
        props.setMaxNumberImages(newValue)
    }
    
    function handleFollowUpOnChange(_, newValue){
      if (!Array.isArray(newValue)) {
        return;
      }
    
      if (newValue[0] !== props.followUpInterval[0]) {
        props.setFollowUpInterval([Math.min(newValue[0], props.followUpInterval[1]), props.followUpInterval[1]]);
      } else {
        props.setFollowUpInterval([props.followUpInterval[0], Math.max(newValue[1], props.followUpInterval[0])]);
      }
    }
      function handleDiseasesChange (event){
        const {
          target: { value },
        } = event;
    
        // On autofill we get a stringified value.
        let newValue = typeof value === 'string' ? value.split(',') : value
    
    
        if ((props.diseasesFilter).indexOf("All") > -1 && newValue.length > 0){
            newValue = diseases.filter((disease) => newValue.indexOf(disease) == -1)
        }
        newValue = newValue.indexOf("All") > -1 ? ["All"] : newValue    
        props.setDiseasesFilter(newValue);
        
      };

    return (
        <div className="filtersContainer">
            <div className="filterContainer">
                <div className="filterTitleContainer">
                    <BsIntersect className="filterIcon"/>
                    <div className="filterName">
                        Similarity threshold
                    </div>
                </div>
                <ThemeProvider theme={sliderTheme}>
                <Slider
                    aria-label="Similarity"
                    value={props.similarityThreshold}
                    step={1}
                    valueLabelDisplay="auto"
                    onChange={handleSimilarityOnChange}
                />
                </ThemeProvider>
            </div>


            <div className="filterContainer">
                <div className="filterTitleContainer">
                    <IoMdImages className="filterIcon"/>
                    <div className="filterName">
                        Maximum number of images
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
                    <AiOutlineOrderedList className="filterIcon"/>
                    <div className="filterName">
                        Follow up
                    </div>
                </div>
                <ThemeProvider theme={sliderTheme}>
                <Slider
                    value={props.followUpInterval}
                    step={1}
                    min={1}
                    max={10} /*Should check what actual max follow up is in the data*/
                    valueLabelDisplay="auto"
                    onChange={handleFollowUpOnChange}
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
