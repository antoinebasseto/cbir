import React, { useState } from 'react'
import "./sidebar.css"
import {FiUpload} from "react-icons/fi"
import {IoMdSettings} from "react-icons/io"
import {BsImages} from "react-icons/bs"

export default function Sidebar(props) {


    return (
    <div className="sidebar">
        <div className="sidebarWrapper">
            <div className="sidebarMenu"> {/* Only useful if we would add other menus */}
                <h3 className="sidebarTitle">Dashboard</h3>
                <hr className="horizontalLine"/>
                <ul className="sidebarList">
                    <li className= {props.indexActiv===0 ? "sidebarListItem active" : "sidebarListItem"} onClick={props.handleUpload}>
                        <FiUpload className="sidebarIcon"/>
                        Upload
                    </li>
                    <li className={props.indexActiv===1 ? "sidebarListItem active" : "sidebarListItem"}  onClick={props.handleShow}>
                        <BsImages className="sidebarIcon"/>
                        Similar Images
                    </li>
                    <li className={props.indexActiv===2 ? "sidebarListItem active" : "sidebarListItem"}  onClick={props.handleFilter}>
                        <IoMdSettings className="sidebarIcon"/>
                        Filter
                    </li>
                </ul>
            </div>
        </div>
    </div>
    )
}
