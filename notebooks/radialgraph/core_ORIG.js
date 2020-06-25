
    
    function FindLineCircleIntersections(cx, cy, radius, point1, point2) {
      let intersections = []
      let dx, dy, A, B, C, det, t

      dx = point2.x - point1.x;
      dy = point2.y - point1.y;

      A = dx * dx + dy * dy;
      B = 2 * (dx * (point1.x - cx) + dy * (point1.y - cy));
      C = (point1.x - cx) * (point1.x - cx) +
          (point1.y - cy) * (point1.y - cy) -
          radius * radius;

      det = B * B - 4 * A * C;
      if ((A <= 0.0000001) || (det < 0))
      {
        // no intersection
      } else if (det == 0) {
          // One solution.
          t = -B / (2 * A);
    
          intersections.push({
            x: point1.x + t * dx,
            y: point1.y + t * dy
          })

      } else {
          // Two solutions.
          t = ((-B + Math.sqrt(det)) / (2 * A));
          intersections.push({
            x: point1.x + t * dx,
            y: point1.y + t * dy
          })

          t = ((-B - Math.sqrt(det)) / (2 * A));
          intersections.push({
            x: point1.x + t * dx,
            y: point1.y + t * dy
          })
      }
      return intersections
  }


    const container = document.getElementById('cont')
    const SVG_NS = 'http://www.w3.org/2000/svg'

    // create bezier curve path from points
    function strBezier(from, to, control1, control2=null) {
      let _control2 = control2 ? control2 : control1
      return `M${from.x} ${from.y} C ${control1.x} ${control1.y}, ${_control2.x} ${_control2.y}, ${to.x} ${to.y}`
    }


    function refreshGraph(parentEl, dataset, options = {}) {
      const minNodeRadius = 'minNodeRadius' in options ? options.minNodeRadius : 10
      const maxNodeRadius = 'maxNodeRadius' in options ? options.maxNodeRadius : 50
      const minEdgeThickness = 'minEdgeThickness' in options ? options.minEdgeThickness : 1.5
      const maxEdgeThickness = 'maxEdgeThickness' in options ? options.maxEdgeThickness : 80
      const nodePadding = 'nodePadding' in options ? options.nodePadding : (minNodeRadius + maxNodeRadius)
      const graphPadding = 'graphPadding' in options ? options.graphPadding : maxNodeRadius * 2
      const graphMode = 'mode' in options && ['directional', 'mono'].includes(options.mode) ? options.mode : 'mono'
      const defaultNodeColor = 'defaultNodeColor' in options ? options.defaultNodeColor : '#07baae'
      const linkColor = 'linkColor' in options ? options.linkColor : 'grey'
      const inboundLinkColor = 'inboundLinkColor' in options ? options.inboundLinkColor : '#076dd9'
      const outboundLinkColor = 'outboundLinkColor' in options ? options.outboundLinkColor : '#db2612'
      const onEdgeEnter = 'onEdgeEnter' in options ? options.onEdgeEnter : null
      const onEdgeLeave = 'onEdgeLeave' in options ? options.onEdgeLeave : null
      const onEdgeClick = 'onEdgeClick' in options ? options.onEdgeClick : null
      const onNodeEnter = 'onNodeEnter' in options ? options.onNodeEnter : null
      const onNodeLeave = 'onNodeLeave' in options ? options.onNodeLeave : null
      const onNodeClick = 'onNodeClick' in options ? options.onNodeClick : null
      const centerGravity = 'centerGravity' in options ? options.centerGravity : 0.3
      const smoothTransition = 'transition: all 0.2s;'


      const defaultCurveOpacity = 0.1
      const highlightedCurveOpacity = 0.9

      let nodes = {}
      let edges = {}
      // to count the number of link to and from nodes. The keys are node ids
      let linkCounter = {}
      const connectionsToTarget = {} // the key is the 'target node' id
      const connectionsFromSource = {} // the key is the 'source node' id
      const connectionFromSourceToTarget = {} // the key is the edge id
      const drawableEdges = {} // keys are 'nodeId_A nodeId_B' where A and B are in alphabetical order


      // sorting and indexing the nodes and the edges
      for (let i = 0; i < dataset.length; i += 1) {
        let el = dataset[i]

        // nodes and edges must at least have an id prop inside of a data prop
        if (!('data' in el) || ('data' in el && !('id' in el.data))) {
          continue
        }

        // this element is an edge
        if ('source' in el.data && 'target' in el.data) {
          const edgeId = el.data.id
          edges[edgeId] = el
          
          const sourceNodeId = el.data.source
          const targetNodeId = el.data.target
          
          // counting connection on source
          if (!(sourceNodeId in linkCounter)) {
            linkCounter[sourceNodeId] = 0
          }
          linkCounter[sourceNodeId] += 1

          // counting conenction on target
          if (!(targetNodeId in linkCounter)) {
            linkCounter[targetNodeId] = 0
          }
          linkCounter[targetNodeId] += 1

          // adding to the outbound connection dictionnary
          if(!(targetNodeId in connectionsToTarget)) {
            connectionsToTarget[targetNodeId] = []
          }
          connectionsToTarget[targetNodeId].push({from: sourceNodeId, by: edgeId})

          // adding to the inbound connection dictionnary
          if (!(sourceNodeId in connectionsFromSource)) {
            connectionsFromSource[sourceNodeId] = []
          }
          connectionsFromSource[sourceNodeId].push({to: targetNodeId, by: edgeId})
    
          // adding to the edges dictionary
          connectionFromSourceToTarget[edgeId] = {
            from: sourceNodeId,
            to: targetNodeId
          }

          // we want to aggregate the edge connections.
          // If mono mode, we dont care about node order, we just want each pair to be in the same order (alphabetical).
          // If directional mode, we do care of the order and keep it with "source target"
          let drawableEdgesKey = graphMode === 'mono' ? [sourceNodeId, targetNodeId].sort().join(' ') : [sourceNodeId, targetNodeId].join(' ')
          if (!(drawableEdgesKey in drawableEdges)) {
            drawableEdges[drawableEdgesKey] = []
          }
          drawableEdges[drawableEdgesKey].push(el)

        } else 
        
        // this element is a node
        {
          let nodeId = el.data.id
          nodes[nodeId] = el

          // make sure that if a node has 0 connections it's still countain the count 0
          // (just to conformity with the others)
          if (!(nodeId in linkCounter)) {
            linkCounter[nodeId] = 0
          }
        }
      }

      // finding the min and max number of connections. This will determine dot sizes
      let nodeIds = Object.keys(nodes)
      let nodeLabels = {}
      let allNumberOfLinks = Object.values(linkCounter)
      let minNumberOfLinks = Math.min(...allNumberOfLinks) // TODO: compare perf of destructuring vs. for loop
      let maxNumberOfLinks = Math.max(...allNumberOfLinks)

      let nodeRadiuses = {}
      let nodePositions = {}
      let nodeRadiusSum = 0

      // computing what is going to be the radius of the whole graph.
      // Also computing the labels
      for (let i = 0; i < nodeIds.length; i += 1) {
        let nodeId = nodeIds[i]
        let nodeNbConnection = linkCounter[nodeId]
        let radius = minNodeRadius + ((nodeNbConnection - minNumberOfLinks) / (maxNumberOfLinks - minNumberOfLinks)) * (maxNodeRadius - minNodeRadius)
        nodeRadiuses[nodeId] = radius
        nodeRadiusSum += radius

        if ('style' in nodes[nodeId] && 'label' in nodes[nodeId].style && nodes[nodeId].style.label.length > 0) {
          nodeLabels[nodeId] = nodes[nodeId].style.label
        } else {
          nodeLabels[nodeId] = nodeId.split('/').pop().split('#').pop()
        }
      }

      const graphPerimeter = 2 * nodeRadiusSum + nodeIds.length * nodePadding
      const graphRadius = graphPerimeter / (2 * Math.PI)
      // let canvasSize = Math.ceil(graphRadius + 2 * maxNodeRadius + 2 * graphPadding)
      let canvasSize = 2 * Math.ceil(graphRadius + maxNodeRadius + graphPadding)
      const graphCenter = {x: canvasSize / 2, y: canvasSize / 2}

      const svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg") // document.createElement('svg')
      svgEl.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
      svgEl.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink')
      svgEl.setAttribute('height', `${canvasSize}`)
      svgEl.setAttribute('width', `${canvasSize}`)
      svgEl.setAttribute('viewBox', `0 0 ${canvasSize} ${canvasSize}`)
      parentEl.appendChild(svgEl)

      // adding the arrow markers
      let arrowPath = document.createElementNS(SVG_NS, 'path')
      arrowPath.setAttributeNS(null, 'd', 'M 0 0 L 2 1 L 0 2 z')
      
      let arrow = document.createElementNS(SVG_NS, 'marker')
      arrow.setAttributeNS(null, 'id', 'arrow')
      arrow.setAttributeNS(null, 'markerWidth', 6)
      arrow.setAttributeNS(null, 'markerHeight', 6)
      arrow.setAttributeNS(null, 'refX', 0)
      arrow.setAttributeNS(null, 'refY', 1)
      arrow.setAttributeNS(null, 'orient', 'auto-start-reverse')
      arrow.appendChild(arrowPath)

      let defs = document.createElementNS(SVG_NS, 'defs')
      defs.appendChild(arrow)
      svgEl.appendChild(defs)


      // compute node positions
      let perimeterProgress = 0
      for (let i = 0; i < nodeIds.length; i += 1) {
        const nodeId = nodeIds[i]
        const node = nodes[nodeIds[i]]
        perimeterProgress += nodeRadiuses[nodeId]
        const angle = (Math.PI * 2) * (perimeterProgress / graphPerimeter)
        const x = Math.cos(angle) * graphRadius + graphCenter.x
        const y = Math.sin(angle) * graphRadius + graphCenter.y
        nodePositions[nodeId] = {x: x, y: y}
        perimeterProgress += nodeRadiuses[nodeId] + nodePadding
      }












      // Adding the svg nodes
      let svgCircleById = {}
      let svgLabelById = {}
      let nodeColorById = {}
      for (let i = 0; i < nodeIds.length; i += 1) {
        let nodeId = nodeIds[i]
        let node = nodes[nodeIds[i]]

        let nodeColor = defaultNodeColor

        if ('style' in node && 'background-color' in node.style) {
          nodeColor = node.style['background-color']
        }

        nodeColorById[nodeId] = nodeColor
        
        let circle = document.createElementNS(SVG_NS, 'circle')
        circle.setAttributeNS(null, 'cx', nodePositions[nodeId].x)
        circle.setAttributeNS(null, 'cy', nodePositions[nodeId].y)
        circle.setAttributeNS(null, 'r', nodeRadiuses[nodeId])
        circle.setAttributeNS(null, 'id', nodeId)
        circle.setAttributeNS(null, 'style', `fill: ${nodeColor}; stroke-width: 0px; stroke: ${nodeColor}; transition: all 0.2s; stroke-opacity: 0.3;` )
        svgEl.appendChild(circle)
        svgCircleById[nodeId] = circle

        circle.addEventListener('mouseenter', (evt) => {
          let nodeId = evt.target.id
          document.getElementById(evt.target.id).style['stroke-width'] = `${Math.max(nodeRadiuses[nodeId] / 2, 5)}px`

          svgLabelById[nodeId].style.fill = nodeColor

          if (graphMode === 'directional') {
            // update color of outbound links
            if (nodeId in svgEdgesBySourceId) {
              svgEdgesBySourceId[nodeId].forEach(ed => {
                ed.style.stroke = outboundLinkColor
                ed.style.opacity = highlightedCurveOpacity
              })
            }

            // update color of inbound links
            if (nodeId in svgEdgesByTargetId) {
              svgEdgesByTargetId[nodeId].forEach(ed => {
                ed.style.stroke = inboundLinkColor
                ed.style.opacity = highlightedCurveOpacity
              })
            }

            

          } else if (graphMode === 'mono') {
            // update color of outbound links
            if (nodeId in svgEdgesBySourceId) {
              svgEdgesBySourceId[nodeId].forEach(ed => {
                ed.style.opacity = highlightedCurveOpacity
              })
            }

            // update color of inbound links
            if (nodeId in svgEdgesByTargetId) {
              svgEdgesByTargetId[nodeId].forEach(ed => {
                ed.style.opacity = highlightedCurveOpacity
              })
            }
          }

          if (nodeId in connectionsFromSource) {
            connectionsFromSource[nodeId].forEach(conn => {
              let connectedToNode = svgCircleById[conn.to]
              connectedToNode.style['stroke-width'] = `${Math.max(nodeRadiuses[conn.to] / 2, 5)}px`
              svgLabelById[conn.to].style.fill = nodeColorById[conn.to] 
            })
          }


          if (nodeId in connectionsToTarget) {
            connectionsToTarget[nodeId].forEach(conn => {
              let connectedFromNode = svgCircleById[conn.from]
              connectedFromNode.style['stroke-width'] = `${Math.max(nodeRadiuses[conn.from] / 2, 5)}px`
              svgLabelById[conn.from].style.fill = nodeColorById[conn.from] 
            })
          }


      
          if (typeof onNodeEnter === 'function') {
            onNodeEnter(nodeId, connectionsToTarget[nodeId] || [], connectionsFromSource[nodeId] || [], nodes, edges)
          }

        })
  
        circle.addEventListener('mouseleave', (evt) => {
          let nodeId = evt.target.id
          document.getElementById(evt.target.id).style['stroke-width'] = '0px'

          svgLabelById[nodeId].style.fill = 'black'


          if (graphMode === 'directional') {
            // update color of outbound links
            if (nodeId in svgEdgesBySourceId) {
              svgEdgesBySourceId[nodeId].forEach(ed => {
                ed.style.stroke = linkColor
                ed.style.opacity = defaultCurveOpacity
              })
            }

            // update color of intbound links
            if (nodeId in svgEdgesByTargetId) {
              svgEdgesByTargetId[nodeId].forEach(ed => {
                ed.style.stroke = linkColor
                ed.style.opacity = defaultCurveOpacity
              })
            }

          } else if (graphMode === 'mono') {
            // update color of outbound links
            if (nodeId in svgEdgesBySourceId) {
              svgEdgesBySourceId[nodeId].forEach(ed => {
                ed.style.opacity = defaultCurveOpacity
              })
            }

            // update color of intbound links
            if (nodeId in svgEdgesByTargetId) {
              svgEdgesByTargetId[nodeId].forEach(ed => {
                ed.style.opacity = defaultCurveOpacity
              })
            }
          }

          if (nodeId in connectionsFromSource) {
            connectionsFromSource[nodeId].forEach(conn => {
              let connectedToNode = svgCircleById[conn.to]
              connectedToNode.style['stroke-width'] = '0px'
              svgLabelById[conn.to].style.fill = 'black'
            })
          }

          if (nodeId in connectionsToTarget) {
            connectionsToTarget[nodeId].forEach(conn => {
              let connectedFromNode = svgCircleById[conn.from]
              connectedFromNode.style['stroke-width'] = '0px'
              svgLabelById[conn.from].style.fill = 'black'
            })
          }

          if (typeof onNodeLeave === 'function') {
            let nodeId = evt.target.id
            onNodeLeave(nodeId, connectionsToTarget[nodeId] || [], connectionsFromSource[nodeId] || [], nodes, edges)
          }

        })


        circle.addEventListener('mousedown', (evt) => {
          if (typeof onNodeClick === 'function') {
            onNodeClick(nodeId, connectionsToTarget[nodeId] || [], connectionsFromSource[nodeId] || [], nodes, edges)
          }
        })


        // the label position is along the axis that goes from the center of the graph to the center of the diagonal
        let nodeRadius = nodeRadiuses[nodeId]
        let nodeCenter = nodePositions[nodeId]
        let graphCenterToNodeCenter = {x: nodeCenter.x - graphCenter.x, y: nodeCenter.y - graphCenter.y}
        let graphCenterToNodeCenterNorm = (graphCenterToNodeCenter.x ** 2 + graphCenterToNodeCenter.y ** 2) ** 0.5
        let graphCenterToNodeCenterNormalized = {x: graphCenterToNodeCenter.x / graphCenterToNodeCenterNorm, y: graphCenterToNodeCenter.y / graphCenterToNodeCenterNorm}
        let labelAnchorPosition = {
          x: nodeCenter.x + graphCenterToNodeCenterNormalized.x * (Math.max(nodeRadius + 2, nodeRadius * 1.5)),
          y: nodeCenter.y + graphCenterToNodeCenterNormalized.y * (Math.max(nodeRadius + 2, nodeRadius * 1.5))
        }

        let textAnchor = nodeCenter.x >= graphCenter.x ? 'start' : 'end'
        let labelRotationDeg = Math.atan2(graphCenterToNodeCenterNormalized.y, graphCenterToNodeCenterNormalized.x) * 180 / Math.PI + (nodeCenter.x >= graphCenter.x ? 0 : 180)
        let fontSize = 5 * nodeRadiuses[nodeId] ** 0.4

        let labelSvg = document.createElementNS(SVG_NS, 'text')
        labelSvg.innerHTML = nodeLabels[nodeId]
        labelSvg.setAttributeNS(null, 'id', nodeId)
        labelSvg.setAttributeNS(null, 'x', labelAnchorPosition.x)
        labelSvg.setAttributeNS(null, 'y', labelAnchorPosition.y)
        labelSvg.setAttributeNS(null, 'text-anchor', textAnchor)
        labelSvg.setAttributeNS(null, 'transform', `rotate(${labelRotationDeg}, ${labelAnchorPosition.x}, ${labelAnchorPosition.y})`)
        labelSvg.setAttributeNS(null, 'alignment-baseline', 'middle')
        labelSvg.setAttributeNS(null, 'style', `font-family: sans-serif; font-size: ${fontSize}; cursor: pointer; user-select: none; transition: all 0.2s;`)
        svgEl.appendChild(labelSvg)

        svgLabelById[nodeId] = labelSvg

        labelSvg.addEventListener('mouseenter', (evt) => {
          evt.target.style.fill = nodeColor
          let nodeId = evt.target.id

          let evt2 = new MouseEvent('mouseenter')
          svgCircleById[nodeId].dispatchEvent(evt2)
        })


        labelSvg.addEventListener('mouseleave', (evt) => {
          evt.target.style.fill = 'black'
          let nodeId = evt.target.id

          let evt2 = new MouseEvent('mouseleave')
          svgCircleById[nodeId].dispatchEvent(evt2)
        })


        labelSvg.addEventListener('mousedown', (evt) => {
          let nodeId = evt.target.id
          let evt2 = new MouseEvent('mousedown')
          svgCircleById[nodeId].dispatchEvent(evt2)
        })

      }









      // compute drawable edges thicknesses and draw them
      const smallestLink = Math.min(...Object.values(drawableEdges).map(de => de.length)) // in number of occurence
      const largestLink = Math.max(...Object.values(drawableEdges).map(de => de.length)) // in number of occurence
      const drawableEdgeKeys = Object.keys(drawableEdges)
      const svgEdgesBySourceId = {}
      const svgEdgesByTargetId = {}
      const drawableEdgesThicknesses = {}
      for (let i = 0; i < drawableEdgeKeys.length; i += 1) {
        const drawableEdgeId = drawableEdgeKeys[i]
        const drawableEdgeLength = drawableEdges[drawableEdgeId].length // in number of occurence
        const thicknesses = minEdgeThickness + ((drawableEdgeLength - smallestLink) / (maxNumberOfLinks - smallestLink)) * (maxEdgeThickness - minEdgeThickness)
        drawableEdgesThicknesses[drawableEdgeId] = thicknesses

        const sourceNodeId = drawableEdges[drawableEdgeId][0].data.source
        const sourceNodePosition = nodePositions[sourceNodeId]
        const sourceNodeRadius = nodeRadiuses[sourceNodeId]

        const targetNodeId = drawableEdges[drawableEdgeId][0].data.target
        const targetNodePosition = nodePositions[targetNodeId]
        const targetNodeRadius = nodeRadiuses[targetNodeId]

        // gives 2 point, but we want to keep only one
        let intersectionsWithSourceNode = FindLineCircleIntersections(sourceNodePosition.x, sourceNodePosition.y, sourceNodeRadius * 0.5, sourceNodePosition, targetNodePosition)
        let hitsourceNode = (sourceNodePosition.x - intersectionsWithSourceNode[0].x) ** 2 + (sourceNodePosition.y - intersectionsWithSourceNode[0].y) ** 2 
                        < (sourceNodePosition.x - intersectionsWithSourceNode[1].x) ** 2 + (sourceNodePosition.y - intersectionsWithSourceNode[1].y) ** 2
                        ? intersectionsWithSourceNode[1]
                        : intersectionsWithSourceNode[0]

        // adding just a bit of randomness to prevent similar edges to be on top of each other
        hitsourceNode.x = hitsourceNode.x + Math.random() * (sourceNodeRadius * 0.2 * 2) - sourceNodeRadius * 0.2
        hitsourceNode.y = hitsourceNode.y + Math.random() * (sourceNodeRadius * 0.2 * 2) - sourceNodeRadius * 0.2
        
        let intersectionsWithTargetNode = FindLineCircleIntersections(targetNodePosition.x, targetNodePosition.y, targetNodeRadius * 0.5, targetNodePosition, sourceNodePosition)
        let hittargetNode = (targetNodePosition.x - intersectionsWithTargetNode[0].x) ** 2 + (targetNodePosition.y - intersectionsWithTargetNode[0].y) ** 2 
                        < (targetNodePosition.x - intersectionsWithTargetNode[1].x) ** 2 + (targetNodePosition.y - intersectionsWithTargetNode[1].y) ** 2
                        ? intersectionsWithTargetNode[1]
                        : intersectionsWithTargetNode[0]

        // adding just a bit of randomness to prevent similar edges to be on top of each other
        hittargetNode.x = hittargetNode.x + Math.random() * (targetNodeRadius * 0.2 * 2) - targetNodeRadius * 0.2
        hittargetNode.y = hittargetNode.y + Math.random() * (targetNodeRadius * 0.2 * 2) - targetNodeRadius * 0.2

        let weightPosition = {
          x: graphCenter.x * centerGravity + ((sourceNodePosition.x + targetNodePosition.x) / 2) * (1 - centerGravity),
          y: graphCenter.y * centerGravity + ((sourceNodePosition.y + targetNodePosition.y) / 2) * (1 - centerGravity),
        }

        // create the svg curve
        let curve = document.createElementNS(SVG_NS, 'path')
        curve.setAttributeNS(null, 'd', strBezier(hitsourceNode, hittargetNode, weightPosition) )
        curve.setAttributeNS(null, 'stroke', linkColor)
        // curve.setAttributeNS(null, 'stroke-width', `${0.5 + Math.random() * 2}px`)
        curve.setAttributeNS(null, 'stroke-width', `${thicknesses}px`)
        curve.setAttributeNS(null, 'fill', 'none')
        curve.setAttributeNS(null, 'opacity', defaultCurveOpacity)
        curve.setAttributeNS(null, 'id', drawableEdgeId)
        curve.setAttributeNS(null, 'sourceNodeId', sourceNodeId)
        curve.setAttributeNS(null, 'targetNodeId', targetNodeId)
        curve.setAttributeNS(null, 'style', `${smoothTransition}`)
        // curve.setAttributeNS(null, 'marker-start', 'url(#arrow)')
        // curve.setAttributeNS(null, 'marker-end', 'url(#arrow)')
        svgEl.appendChild(curve)

        if (!(sourceNodeId in svgEdgesBySourceId)) {
          svgEdgesBySourceId[sourceNodeId] = []
        }
        svgEdgesBySourceId[sourceNodeId].push(curve)

        if (!(targetNodeId in svgEdgesByTargetId)) {
          svgEdgesByTargetId[targetNodeId] = []
        }
        svgEdgesByTargetId[targetNodeId].push(curve)


        curve.addEventListener('mouseenter', (evt) => {
          let drawableEdgeId = evt.target.id
          let edge = drawableEdges[drawableEdgeId][0] // the first, all the others are about the same nodes anyways
          let sourceNodeId = edge.data.source
          let targetNodeId = edge.data.target

          svgCircleById[sourceNodeId].style['stroke-width'] = `${Math.max(nodeRadiuses[sourceNodeId] / 2, 5)}px`
          svgCircleById[targetNodeId].style['stroke-width'] = `${Math.max(nodeRadiuses[targetNodeId] / 2, 5)}px`
          svgLabelById[sourceNodeId].style.fill = nodeColorById[sourceNodeId] 
          svgLabelById[targetNodeId].style.fill = nodeColorById[targetNodeId] 


          evt.target.style.opacity = highlightedCurveOpacity

          if (typeof onEdgeEnter === 'function') {
            onEdgeEnter(drawableEdges[drawableEdgeId] || [], nodes, edges)
          }
        })

        curve.addEventListener('mousedown', (evt) => {
          let drawableEdgeId = evt.target.id

          if (typeof onEdgeClick === 'function') {
            onEdgeClick(drawableEdges[drawableEdgeId] || [], nodes, edges)
          }
        })

        curve.addEventListener('mouseleave', (evt) => {
          let drawableEdgeId = evt.target.id
          let edge = drawableEdges[drawableEdgeId][0] // the first, all the others are about the same nodes anyways
          let sourceNodeId = edge.data.source
          let targetNodeId = edge.data.target

          svgCircleById[sourceNodeId].style['stroke-width'] = '0px'
          svgCircleById[targetNodeId].style['stroke-width'] = '0px'
          svgLabelById[sourceNodeId].style.fill = 'black'
          svgLabelById[targetNodeId].style.fill = 'black'

          evt.target.style.opacity = defaultCurveOpacity

          if (typeof onEdgeLeave === 'function') {
            onEdgeLeave(drawableEdges[drawableEdgeId]  || [], nodes, edges)
          }
        })
        
      }


      
    }

